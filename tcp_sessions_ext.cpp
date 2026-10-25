/**
 * tcp_sessions_ext.cpp
 *
 *
 * Usage (Python):
 *   from tcp_sessions_ext import TCPSession, TCPSessionExtractor
 *
 * Build:
 *   python3 setup.py build_ext --inplace
 *
 * Memory model:
 *   __iter__ / extract() stream sessions one-by-one as they close.  Peak
 *   memory is O(concurrent active sessions), not O(all sessions in the file).
 *   For a 30 GB PCAP this is typically < 1 GB instead of 75 GB.
 *
 *   Optional knob: max_packets_per_session (default 0 = unlimited).  Packets
 *   beyond the cap are still tracked for lifecycle (FIN/RST/duration) but
 *   their bytes are discarded immediately, further bounding per-session memory.
 *
 * Notes:
 *   - Live capture uses libpcap directly (no scapy / AsyncSniffer needed).
 *   - show_progress accepted for API compatibility; currently ignored.
 *   - BPF filters applied for live capture only; file-mode TCP filtering is
 *     done in the parser (macOS system libpcap 1.0 offline BPF compat issue).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <pcap.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

// Ethernet protocol type constants (BSD/macOS compat)
#ifndef ETH_P_IP
#define ETH_P_IP       0x0800
#endif
#ifndef ETH_P_8021Q
#define ETH_P_8021Q    0x8100
#endif
#ifndef ETH_P_8021AD
#define ETH_P_8021AD   0x88a8
#endif


template <typename T>
inline void hash_combine(std::size_t& seed, const T& value)
{
    constexpr std::size_t magic =
        sizeof(std::size_t) == 8
            ? 0x9e3779b97f4a7c15ULL
            : 0x9e3779b9U;

    seed ^= std::hash<T>{}(value)
          + magic
          + (seed << 6)
          + (seed >> 2);
}

// ---------------------------------------------------------------------------
// IP address — unified 4/16-byte representation
// ---------------------------------------------------------------------------
struct IPAddr {
    uint8_t b[16]{};
    uint8_t len{4};

    IPAddr() = default;
    IPAddr(const uint8_t* data, uint8_t l) : len(l) { std::memcpy(b, data, l); }

    bool operator==(const IPAddr& o) const {
        return len == o.len && std::memcmp(b, o.b, len) == 0;
    }

    bool operator<(const IPAddr& o) const {
        int cmp = std::memcmp(b, o.b, sizeof(b));
        if (cmp != 0)
        {
            return cmp < 0;
        }
        return len < o.len;
    }

    std::string to_string() const {
        char buf[INET6_ADDRSTRLEN]{};
        inet_ntop(len == 4 ? AF_INET : AF_INET6, b, buf, sizeof(buf));
        return buf;
    }
};

template<>
struct std::hash<IPAddr>
{
    std::size_t operator()(const IPAddr& ip) const noexcept
    {
        size_t h = 0;
        hash_combine(h, ip.len);
        for (uint8_t i = 0; i < ip.len; ++i)
        {
            hash_combine(h, ip.b[i]);
        }
        return h;
    }
};

struct TCPEndpoint
{
    TCPEndpoint() = default;
    TCPEndpoint(const IPAddr& ip_, uint16_t port_):
    ip(ip_),
    port(port_)
    {}

    IPAddr ip;
    uint16_t port;

    bool operator==(const TCPEndpoint& o) const {
        return std::tie(ip, port) == std::tie(o.ip, o.port);
    }

    bool operator<(const TCPEndpoint& o) const {
        return std::tie(ip, port) < std::tie(o.ip, o.port);
    }
};

template<>
struct std::hash<TCPEndpoint>
{
    std::size_t operator()(const TCPEndpoint& endpoint) const noexcept
    {
        size_t h = 0;
        hash_combine(h, endpoint.ip);
        hash_combine(h, endpoint.port);
        return h;
    }
};

// ---------------------------------------------------------------------------
// Session key
// ---------------------------------------------------------------------------
struct SessionKey {
    TCPEndpoint src;
    TCPEndpoint dst;

    bool operator==(const SessionKey& o) const {
        return std::tie(src, dst) == std::tie(o.src, o.dst);
    }
};

template<>
struct std::hash<SessionKey>
{
    std::size_t operator()(const SessionKey& k) const noexcept
    {
        size_t hv = 0;
        hash_combine(hv, std::hash<TCPEndpoint>()(k.src));
        hash_combine(hv, std::hash<TCPEndpoint>()(k.dst));
        return hv;
    }
};

class BidirectionalSessionKey {
public:
    BidirectionalSessionKey(const SessionKey& session_key):
    BidirectionalSessionKey(session_key.src, session_key.dst)
    {}

    BidirectionalSessionKey(const TCPEndpoint& endpoint_a, const TCPEndpoint& endpoint_b):
    lesser_(std::min(endpoint_a, endpoint_b)),
    greater_(std::max(endpoint_a, endpoint_b))
    {
    }

    bool operator==(const BidirectionalSessionKey& o) const {
        return std::tie(lesser_, greater_) == std::tie(o.lesser_, o.greater_);
    }

private:
    TCPEndpoint lesser_;
    TCPEndpoint greater_;

    friend struct std::hash<BidirectionalSessionKey>;
};

template<>
struct std::hash<BidirectionalSessionKey>
{
    std::size_t operator()(const BidirectionalSessionKey& k) const noexcept
    {
        size_t hv = 0;
        hash_combine(hv, std::hash<TCPEndpoint>()(k.lesser_));
        hash_combine(hv, std::hash<TCPEndpoint>()(k.greater_));
        return hv;
    }
};

// ---------------------------------------------------------------------------
// Per-packet dedup key
// ---------------------------------------------------------------------------
struct DedupKey {
    uint8_t  direction{};
    uint32_t seq{};
    uint32_t payload_len{};

    bool operator==(const DedupKey& o) const {
        return direction == o.direction && seq == o.seq && payload_len == o.payload_len;
    }
};

template<>
struct std::hash<DedupKey>
{
    std::size_t operator()(const DedupKey& k) const noexcept
    {
        size_t h = 0;
        hash_combine(h, k.direction);
        hash_combine(h, k.seq);
        hash_combine(h, k.payload_len);
        return h;
    }
};

using DedupSet = std::unordered_set<DedupKey>;

// ---------------------------------------------------------------------------
// TCPSession
// ---------------------------------------------------------------------------
struct PacketRecord {
    double      timestamp{};
    std::vector<uint8_t> payload;
    std::vector<uint8_t> tls_record_content_types;
};

struct TCPSession {
    TCPEndpoint client;
    TCPEndpoint server;
    double      start_time{};
    double      end_time{-1.0};
    bool        ended_with_fin{false};
    bool        ended_with_rst{false};

    std::vector<PacketRecord> client_to_server_packets;
    std::vector<PacketRecord> server_to_client_packets;

    double duration()      const { return end_time < 0 ? 0.0 : end_time - start_time; }
    size_t    total_packets() const {
        return client_to_server_packets.size() + server_to_client_packets.size();
    }
};

// ---------------------------------------------------------------------------
// TLS header check
// ---------------------------------------------------------------------------
static bool is_valid_tls_record_header(const uint8_t* payload, size_t payload_len) {
    if (payload_len < 5) return false;
    const uint8_t ct  = static_cast<uint8_t>(payload[0]);
    if (ct != 0x14 && ct != 0x15 && ct != 0x16 && ct != 0x17) return false;
    const uint16_t ver = (static_cast<uint8_t>(payload[1]) << 8) |
                          static_cast<uint8_t>(payload[2]);
    return ver >= 0x0301 && ver <= 0x0304;
}

static bool is_valid_tls_header(const std::vector<uint8_t>& p) {
    return is_valid_tls_record_header(p.data(), p.size());
}

static std::vector<uint8_t> parse_tls_record_content_types(const uint8_t* payload,
                                                           size_t payload_len) {
    std::vector<uint8_t> content_types;
    size_t offset = 0;
    while ((offset + 5) <= payload_len) {
        const uint8_t* header = payload + offset;
        const size_t remaining = payload_len - offset;
        if (!is_valid_tls_record_header(header, remaining)) {
            break;
        }

        const uint16_t record_len =
            (static_cast<uint16_t>(header[3]) << 8) | static_cast<uint16_t>(header[4]);
        content_types.push_back(header[0]);

        const size_t record_total_len = 5 + static_cast<size_t>(record_len);
        if (record_total_len > remaining) {
            break;
        }
        offset += record_total_len;
    }
    return content_types;
}

// ---------------------------------------------------------------------------
// ExtractorState
//
// `completed` is a small staging buffer — the iterator drains it after every
// batch of packets.  It never accumulates more than the sessions that closed
// during one BATCH_SIZE burst, so it stays small.
// ---------------------------------------------------------------------------
struct ExtractorState {
    // Hard wall-clock lifetime cap (closes the session). When the next
    // packet's timestamp is >= start_time + this value, the session is
    // closed with end_time = start_time + cap and yielded
    // (stats["closed_by_duration"]). The triggering packet is NOT added
    // to the session. std::nullopt = unlimited (Python: None or 0).
    std::optional<double> max_session_duration_sec;
    // Idle/staleness cap (closes the session). Every STALE_SWEEP_INTERVAL
    // packets, any open session whose start_time is older than
    // (current_ts - this value) is purged and yielded
    // (stats["purged_stale"]). Acts as a safety net for sessions that
    // never receive FIN/RST and don't hit max_session_duration_sec.
    // std::nullopt = disabled (Python: None or 0).
    std::optional<double> stale_session_timeout_sec;
    // Per-direction storage cap (memory bound only — does NOT close the session).
    // Once client_to_server_packets or server_to_client_packets reaches this
    // size, further payloads in that direction are discarded (bytes thrown
    // away); the session keeps running and is still closed by FIN/RST,
    // duration, stale purge, EOF, or max_session_packets. 0 = unlimited.
    size_t                max_packets_per_session{0};
    // Total-packet lifecycle cap (closes the session). Once the sum of
    // stored packets across both directions reaches this value, the session
    // is closed with end_time = current packet ts and yielded
    // (stats["closed_by_packet_count"]). 0 = unlimited.
    size_t                max_session_packets{0};
    // Minimum number of stored payload packets required in EACH direction
    // before yielding a session. Sessions below this threshold are dropped.
    size_t                min_packets_per_direction{2};

    // Throttle full open-session sweeps to limit O(N) per-packet cost.
    size_t packets_since_sweep{0};
    static constexpr int STALE_SWEEP_INTERVAL = 256;

    std::unordered_map<SessionKey, TCPSession> sessions;
    std::unordered_map<BidirectionalSessionKey, SessionKey> session_index;
    std::unordered_map<SessionKey, DedupSet> seen_packets;

    std::vector<TCPSession> completed;  // drained by iterator after each burst

    struct statistics {
        size_t started_on_syn_ack = 0;
        size_t closed_by_rst_fin = 0;
        size_t closed_by_duration = 0;
        size_t closed_by_packet_count = 0;
        size_t purged_stale = 0;
        size_t flushed_at_eof = 0;
        size_t dropped_no_significant_payload = 0;
        size_t dropped_invalid_tls = 0;
        size_t dropped_session_not_found = 0;
        size_t packets_processed = 0;
        size_t sessions_yielded = 0;
    };
    statistics stats;

    static std::map<std::string, size_t> to_map(const statistics& stats)
    {
        return std::map<std::string, size_t>{
                                          {"started_on_syn_ack",             stats.started_on_syn_ack},
                                          {"closed_by_rst_fin",              stats.closed_by_rst_fin},
                                          {"closed_by_duration",             stats.closed_by_duration},
                                          {"closed_by_packet_count",         stats.closed_by_packet_count},
                                          {"purged_stale",                   stats.purged_stale},
                                          {"flushed_at_eof",                 stats.flushed_at_eof},
                                          {"dropped_no_significant_payload", stats.dropped_no_significant_payload},
                                          {"dropped_invalid_tls",            stats.dropped_invalid_tls},
                                          {"dropped_session_not_found",      stats.dropped_session_not_found},
                                          {"packets_processed",              stats.packets_processed},
                                          {"sessions_yielded",               stats.sessions_yielded},
                                      };
    }

    void close_session(const SessionKey& key) {
        auto it = sessions.find(key);
        if (it == sessions.end()) return;

        TCPSession sess = std::move(it->second);
        sessions.erase(it);
        seen_packets.erase(key);
        session_index.erase(BidirectionalSessionKey(key));

        // Ensure end_time is set so duration is meaningful for sessions
        // closed by EOF flush or stale purge (no FIN/RST, no duration trigger).
        if (sess.end_time < 0) {
            double last_ts = sess.start_time;
            if (!sess.client_to_server_packets.empty())
                last_ts = std::max(last_ts, sess.client_to_server_packets.back().timestamp);
            if (!sess.server_to_client_packets.empty())
                last_ts = std::max(last_ts, sess.server_to_client_packets.back().timestamp);
            sess.end_time = last_ts;
        }

        if (sess.client_to_server_packets.size() < min_packets_per_direction ||
            sess.server_to_client_packets.size() < min_packets_per_direction) {
            stats.dropped_no_significant_payload++;
            return;
        }

        bool tls_ok = false;
        for (const auto* vec : {&sess.client_to_server_packets,
                                 &sess.server_to_client_packets}) {
            for (size_t i = 0; i < vec->size() && !tls_ok; ++i)
            {
                tls_ok = is_valid_tls_header((*vec)[i].payload);
            }
        }
        if (!tls_ok) {
            stats.dropped_invalid_tls++; return;
        }

        stats.sessions_yielded++;
        completed.push_back(std::move(sess));
    }

    struct vlan_tag {
        uint16_t tci;
        uint16_t encapsulated_proto;
    } __attribute__((packed));


    const struct ip* find_ipv4_header(const uint8_t* packet,
                                    size_t packet_len)
    {
        if (packet_len < sizeof(ether_header))
        {
            return nullptr;
        }

        auto* eth =
            reinterpret_cast<const ether_header*>(packet);

        uint16_t ether_type = ntohs(eth->ether_type);

        const uint8_t* ptr = packet + sizeof(ether_header);
        size_t remaining = packet_len - sizeof(ether_header);

        // Skip any number of VLAN tags
        while (ether_type == ETH_P_8021Q ||  // 802.1Q
            ether_type == ETH_P_8021AD)    // 802.1ad
        {
            if (remaining < sizeof(vlan_tag))
            {
                return nullptr;
            }

            auto* vlan =
                reinterpret_cast<const vlan_tag*>(ptr);

            ether_type = ntohs(vlan->encapsulated_proto);

            ptr += sizeof(vlan_tag);
            remaining -= sizeof(vlan_tag);
        }

        if (ether_type != ETH_P_IP)
        {
            return nullptr;
        }

        if (remaining < sizeof(struct ip))
        {
            return nullptr;
        }

        return reinterpret_cast<const struct ip*>(ptr);
    }

    void process_packet(double ts, const uint8_t* buf, uint32_t caplen) {
        stats.packets_processed++;

        // Periodically sweep open sessions whose lifetime has exceeded
        // stale_session_timeout_sec without a closing event.
        if (stale_session_timeout_sec.has_value() && !sessions.empty()) {
            if (++packets_since_sweep >= STALE_SWEEP_INTERVAL) {
                packets_since_sweep = 0;
                purge_stale_sessions(ts);
            }
        }

        const ip* ip = find_ipv4_header(buf, caplen);
        if (!ip)
        {
            return;
        }

        if (ip->ip_v != 4){
            return;  // IPv4 only
        }

        const uint16_t ip_total = ntohs(ip->ip_len);
        const uint8_t ip_header_len = ip->ip_hl * 4;

        if (ip->ip_p != IPPROTO_TCP){
            return; // TCP only
        }

        const uint8_t* packet_end = buf + caplen;
        const uint8_t* tcp_ptr =
            reinterpret_cast<const uint8_t*>(ip) + ip_header_len;
        if (tcp_ptr + sizeof(struct tcphdr) > packet_end)
        {
            // Not enough data for TCP header.
            return;
        }
        const struct tcphdr* tcp = reinterpret_cast<const struct tcphdr*>(tcp_ptr);

        IPAddr src_ip(reinterpret_cast<const uint8_t*>(&ip->ip_src.s_addr), sizeof(ip::ip_src));
        IPAddr dst_ip(reinterpret_cast<const uint8_t*>(&ip->ip_dst.s_addr), sizeof(ip::ip_dst));
        const uint16_t sport   = ntohs(tcp->th_sport);
        const uint16_t dport   = ntohs(tcp->th_dport);
        const uint32_t seq     = ntohl(tcp->th_seq);
        const uint8_t data_off = tcp->th_off * 4;
        const uint8_t* tcp_data_ptr = tcp_ptr + data_off;
        if (data_off < sizeof(struct tcphdr) || tcp_data_ptr > packet_end)
        {
            return;
        }

        const uint8_t flags = tcp->th_flags;

        const int declared    = int(ip_total) - int(ip_header_len) - int(data_off);
        const size_t avail    = (caplen > (tcp_ptr - buf) + data_off)
                                    ? (caplen - (tcp_ptr - buf) - data_off) : 0;
        const int payload_len = std::max(0, std::min(declared, int(avail)));

        // ---- SYN-ACK: create session -----------------------------------
        if ((flags & TH_SYN) && (flags & TH_ACK)) {
            SessionKey canon;
            canon.src = TCPEndpoint(dst_ip, dport);
            canon.dst = TCPEndpoint(src_ip, sport);
            if (sessions.find(canon) == sessions.end()) {
                TCPSession s;
                s.client       = TCPEndpoint(dst_ip, dport);
                s.server       = TCPEndpoint(src_ip, sport);
                s.start_time      = ts;
                sessions[canon]      = std::move(s);
                seen_packets[canon]  = DedupSet{};
                session_index[BidirectionalSessionKey(canon)] = canon;
                stats.started_on_syn_ack++;
            }
        }

        // ---- Session lookup --------------------------------------------
        auto idx_it = session_index.find(BidirectionalSessionKey(TCPEndpoint(src_ip, sport), TCPEndpoint(dst_ip, dport)));
        if (idx_it == session_index.end()) {
            stats.dropped_session_not_found++;
            return;
        }
        const SessionKey sess_key = idx_it->second;
        TCPSession& sess = sessions[sess_key];

        const uint8_t direction = (src_ip == sess.client.ip) ? 0 : 1;

        // ---- Enforce max session duration BEFORE appending this packet --
        // Cap is a hard bound relative to start_time; triggering packet is
        // not added to the session.
        if (max_session_duration_sec.has_value() &&
            (ts - sess.start_time) >= *max_session_duration_sec) {
            sess.end_time = sess.start_time + *max_session_duration_sec;
            stats.closed_by_duration++;
            close_session(sess_key);
            return;
        }

        // ---- Payload dedup + storage -----------------------------------
        if (payload_len > 0) {
            DedupKey dk{direction, seq, uint32_t(payload_len)};
            auto& seen = seen_packets[sess_key];
            if (seen.insert(dk).second) {  // true = newly inserted (not a dupe)
                auto& pkt_vec = (direction == 0) ? sess.client_to_server_packets
                                                  : sess.server_to_client_packets;
                // Discard bytes beyond cap; packet is still in dedup set so
                // we don't double-count retransmits.
                if (max_packets_per_session == 0 ||
                    pkt_vec.size() < max_packets_per_session) {
                    std::vector<uint8_t> payload{tcp_data_ptr, tcp_data_ptr + payload_len};
                    std::vector<uint8_t> content_types =
                        parse_tls_record_content_types(payload.data(), payload.size());
                    pkt_vec.push_back({ts, std::move(payload), std::move(content_types)});
                }
            }
        }

        // ---- Session termination on total packet count ----------------
        // Close once the session has accumulated max_session_packets total
        // stored packets across both directions. Uses stored counts so that
        // pure retransmits (deduped) and beyond-cap discards don't count.
        if (max_session_packets > 0 &&
            (sess.client_to_server_packets.size() +
             sess.server_to_client_packets.size()) >= max_session_packets) {
            sess.end_time = ts;
            stats.closed_by_packet_count++;
            close_session(sess_key);
            return;
        }

        // ---- Session termination on FIN/RST ----------------------------
        if (flags & 0x04) {  // RST
            sess.ended_with_rst = true;  sess.end_time = ts;
            stats.closed_by_rst_fin++;
            close_session(sess_key);
            return;
        }
        if (flags & 0x01) {  // FIN
            sess.ended_with_fin = true;  sess.end_time = ts;
            stats.closed_by_rst_fin++;
            close_session(sess_key);
        }
    }

    void purge_stale_sessions(double current_ts) {
        if (!stale_session_timeout_sec.has_value() || sessions.empty())
        {
            return;
        }

        const double cutoff = current_ts - *stale_session_timeout_sec;
        std::vector<SessionKey> stale_keys;
        stale_keys.reserve(sessions.size());
        // Collect all the stale keys so their sessions can be closed.
        // It's necessary to copy them first because close_session modifies sessions.
        for (const auto& [k, s] : sessions) {
            if (s.start_time <= cutoff) stale_keys.push_back(k);
        }
        for (const auto& key : stale_keys) {
            auto it = sessions.find(key);
            if (it == sessions.end())
            {
                 continue;
            }
            if (max_session_duration_sec.has_value()) {
                it->second.end_time =
                    it->second.start_time + *max_session_duration_sec;
            } else {
                it->second.end_time = current_ts;
            }
            stats.purged_stale++;
            close_session(key);
        }
    }

    void flush_all() {
        std::vector<SessionKey> keys;
        keys.reserve(sessions.size());
        for (auto& [k, _] : sessions)
        {
            keys.push_back(k);
        }
        for (auto& k : keys)
        {
             stats.flushed_at_eof++;
             close_session(k);
        }
    }
};

// ---------------------------------------------------------------------------
// TCPSessionIterator
//
// Each call to try_next() reads packets until a session closes (or EOF).
// Returned sessions are immediately removed from internal storage, so heap
// usage stays proportional to active-session count, not file size.
// ---------------------------------------------------------------------------
static constexpr int BATCH_SIZE = 2048;

class TCPSessionIterator {
public:
    TCPSessionIterator(pcap_t*               handle,
                       ExtractorState         state,
                       bool                   is_live,
                       std::optional<size_t>     live_packet_count,
                       std::optional<double>  live_timeout)
        : handle_(handle), state_(std::move(state)), is_live_(is_live),
          live_packet_count_(live_packet_count), live_timeout_(live_timeout),
          started_(std::chrono::steady_clock::now())
    {}

    ~TCPSessionIterator() {
        if (handle_) {
            pcap_close(handle_); handle_ = nullptr;
        }
    }

    TCPSessionIterator(const TCPSessionIterator&)            = delete;
    TCPSessionIterator& operator=(const TCPSessionIterator&) = delete;

    // Returns empty optional when iteration is complete (no Python exceptions set).
    // This is safe to call from both C++ and from the nanobind __next__ binding.
    std::optional<TCPSession> try_next() {
        // 1. Drain buffered sessions from the last batch
        if (buf_idx_ < buf_.size()) {
            std::optional<TCPSession> s = std::move(buf_[buf_idx_++]);
            if (buf_idx_ == buf_.size())
            {
                buf_.clear();
                buf_idx_ = 0;
            }
            return s;
        }

        if (eof_)
        {
            return std::nullopt;
        }

        // 2. Read packets in batches until at least one session closes
        while (buf_.empty() && !eof_) {
            for (int i = 0; i < BATCH_SIZE && !eof_; ++i) {
                if (!read_one_packet())
                {
                    break;
                }
            }
            // Move completed sessions into buf_ without any Python API calls
            std::swap(buf_, state_.completed);
            state_.completed.clear();
            buf_idx_ = 0;
        }

        if (buf_.empty())
        {
            return std::nullopt;
        }

        std::optional<TCPSession> s = std::move(buf_[buf_idx_++]);
        if (buf_idx_ == buf_.size())
        {
            buf_.clear(); buf_idx_ = 0;
        }
        return s;
    }

    const ExtractorState::statistics& stats() const { return state_.stats; }

private:
    pcap_t*        handle_{};
    ExtractorState state_;
    bool           is_live_{false};

    std::optional<size_t>  live_packet_count_;
    std::optional<double>  live_timeout_;
    std::chrono::steady_clock::time_point started_;

    bool                    eof_{false};
    size_t                  packets_seen_{0};
    std::vector<TCPSession> buf_;
    size_t                  buf_idx_{0};

    // Returns false when we should stop reading
    bool read_one_packet() {
        if (is_live_) {
            if (live_packet_count_.has_value() &&
                packets_seen_ >= *live_packet_count_) {
                state_.flush_all();
                eof_ = true;
                return false;
            }
            if (live_timeout_.has_value()) {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_).count();
                if (elapsed >= *live_timeout_) {
                    state_.flush_all();
                    eof_ = true;
                    return false;
                }
            }
        }

        struct pcap_pkthdr* hdr{};
        const uint8_t*      data{};
        int ret = pcap_next_ex(handle_, &hdr, &data);

        if (ret == 1) {
            if (hdr && data) {
                state_.process_packet(
                    hdr->ts.tv_sec + hdr->ts.tv_usec / 1e6, data, hdr->caplen);
                ++packets_seen_;
            }
            return true;
        }
        if (ret == 0)
        {
            return true;  // live read timeout — re-check conditions
        }

        // ret == -2: EOF (file); ret == -1: error
        state_.flush_all();
        eof_ = true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// TCPSessionExtractor — public Python-facing class
// ---------------------------------------------------------------------------
class TCPSessionExtractor {
public:
    std::optional<std::string> pcap_path;
    std::optional<std::string> iface;
    bool                        show_progress;
    std::string                 bpf_filter;
    std::optional<double>       live_timeout;
    std::optional<size_t>          live_packet_count;
    // Hard wall-clock lifetime cap; closes & yields the session when the
    // next packet's ts exceeds start_time + this value. std::nullopt =
    // unlimited. See ExtractorState for details.
    std::optional<double>       max_session_duration_sec;
    // Idle/staleness cap; purges open sessions older than this on a
    // periodic sweep. std::nullopt = disabled. See ExtractorState.
    std::optional<double>       stale_session_timeout_sec;
    // Per-direction storage cap; memory bound only, never closes a session.
    // 0 = unlimited. See ExtractorState for details.
    size_t                      max_packets_per_session;
    // Total-packet lifecycle cap; closes & yields the session once the
    // combined stored packet count (both directions) reaches this value.
    // 0 = unlimited. See ExtractorState for details.
    size_t                      max_session_packets;
    // Minimum stored payload packets required in EACH direction to yield.
    // Sessions below this threshold are dropped. Default keeps prior behavior.
    size_t                      min_packets_per_direction;

    // Tracks the most recently created iterator so .stats reflects it.
    std::shared_ptr<TCPSessionIterator> last_iter_;

    TCPSessionExtractor(
        nb::object  pcap_path_obj,
        nb::object  interface_obj,
        bool        show_progress_,
        std::string bpf_filter_,
        nb::object  live_timeout_obj,
        nb::object  live_packet_count_obj,
        nb::object  max_session_duration_obj,
        nb::object  stale_session_timeout_obj,
        size_t      max_packets_per_session_,
                size_t      max_session_packets_,
                size_t      min_packets_per_direction_)
        : show_progress(show_progress_), bpf_filter(std::move(bpf_filter_)),
          max_packets_per_session(max_packets_per_session_),
                    max_session_packets(max_session_packets_),
                    min_packets_per_direction(min_packets_per_direction_)
    {
        if (!pcap_path_obj.is_none())
        {
            pcap_path = nb::cast<std::string>(pcap_path_obj);
        }
        if (!interface_obj.is_none())
        {
            iface = nb::cast<std::string>(interface_obj);
        }
        if (!live_timeout_obj.is_none())
        {
            live_timeout = nb::cast<double>(live_timeout_obj);
        }
        if (!live_packet_count_obj.is_none())
        {
            live_packet_count = nb::cast<size_t>(live_packet_count_obj);
        }
        if (!max_session_duration_obj.is_none())
        {
            max_session_duration_sec = nb::cast<double>(max_session_duration_obj);
        }
        if (!stale_session_timeout_obj.is_none())
        {
            stale_session_timeout_sec = nb::cast<double>(stale_session_timeout_obj);
        }

        // Treat non-positive values the same as None (= unlimited / disabled)
        // for symmetry with the size-based knobs (where 0 means unlimited).
        if (max_session_duration_sec.has_value() && *max_session_duration_sec <= 0.0)
        {
            max_session_duration_sec.reset();
        }
        if (stale_session_timeout_sec.has_value() && *stale_session_timeout_sec <= 0.0)
        {
            stale_session_timeout_sec.reset();
        }

        if (pcap_path.has_value() == iface.has_value())
        {
            throw std::invalid_argument("Provide exactly one of pcap_path or interface");
        }
    }

    // Creates a fresh streaming iterator.  Each call opens the source from scratch.
    std::shared_ptr<TCPSessionIterator> make_iterator() {
        auto [handle, is_live] = open_source();
        ExtractorState state;
        state.max_session_duration_sec = max_session_duration_sec;
        state.stale_session_timeout_sec = stale_session_timeout_sec;
        state.max_packets_per_session  = max_packets_per_session;
        state.max_session_packets      = max_session_packets;
        state.min_packets_per_direction = min_packets_per_direction;
        last_iter_ = std::make_shared<TCPSessionIterator>(
            handle, std::move(state), is_live, live_packet_count, live_timeout);
        return last_iter_;
    }

    // Collect all sessions into a std::vector (nanobind/stl converts to Python list).
    // Internally streams so memory stays bounded — only the result list is large.
    std::vector<TCPSession> extract() {
        std::vector<TCPSession> result;
        auto it = make_iterator();  // sets last_iter_
        while (auto opt = it->try_next()) {
            result.push_back(std::move(*opt));
        }
        return result;
    }

private:
    std::pair<pcap_t*, bool> open_source() {
        char errbuf[PCAP_ERRBUF_SIZE]{};

        if (pcap_path.has_value()) {
            pcap_t* h = pcap_open_offline(pcap_path->c_str(), errbuf);
            if (!h)
            {
                throw std::runtime_error(
                std::string("pcap_open_offline: ") + errbuf);
            }
            // BPF skipped for file mode — see file header comment
            return {h, false};
        }

        pcap_t* h = pcap_open_live(iface->c_str(), 65535, 1, 100, errbuf);
        if (!h)
        {
            throw std::runtime_error(
            std::string("pcap_open_live: ") + errbuf);
        }
        apply_bpf(h, bpf_filter);
        return {h, true};
    }

    static void apply_bpf(pcap_t* handle, const std::string& filter) {
        if (filter.empty()) return;
        struct bpf_program fp{};
        if (pcap_compile(handle, &fp, filter.c_str(), 1, 0) < 0) {
            std::string msg = std::string("pcap_compile: ") + pcap_geterr(handle);
            pcap_close(handle); throw std::runtime_error(msg);
        }
        if (pcap_setfilter(handle, &fp) < 0) {
            std::string msg = std::string("pcap_setfilter: ") + pcap_geterr(handle);
            pcap_freecode(&fp); pcap_close(handle); throw std::runtime_error(msg);
        }
        pcap_freecode(&fp);
    }
};

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------
NB_MODULE(tcp_sessions_ext, m) {
    m.doc() =
        "Memory-efficient C++ TCP session extractor (nanobind).\n\n"
        "Sessions stream one-by-one as they close: peak memory is proportional\n"
        "to concurrent active sessions, not total PCAP size.\n\n"
        "  from tcp_sessions_ext import TCPSession, TCPSessionExtractor\n"
        "  for sess in TCPSessionExtractor(pcap_path='cap.pcap'):\n"
        "      process(sess)\n\n"
        "  # or\n"
        "  sessions = TCPSessionExtractor(pcap_path='cap.pcap').extract()";

    nb::class_<TCPSession>(m, "TCPSession")
        .def_prop_ro("client_ip",
            [](const TCPSession& s){ return s.client.ip.to_string(); })
        .def_prop_ro("client_port", [](const TCPSession& s){ return s.client.port; })
        .def_prop_ro("server_ip",
            [](const TCPSession& s){ return s.server.ip.to_string(); })
        .def_prop_ro("server_port", [](const TCPSession& s){ return s.server.port; })
        .def_ro("start_time",     &TCPSession::start_time)
        .def_prop_ro("end_time", [](const TCPSession& s) -> nb::object {
            if (s.end_time < 0) return nb::none();
            return nb::float_(s.end_time);
        })
        .def_ro("ended_with_fin", &TCPSession::ended_with_fin)
        .def_ro("ended_with_rst", &TCPSession::ended_with_rst)
        .def_prop_ro("duration",
            [](const TCPSession& s) { return s.duration(); })
        .def_prop_ro("total_packets",
            [](const TCPSession& s) { return s.total_packets(); })
        .def_prop_ro("client_to_server_packets", [](const TCPSession& s) {
            nb::list lst;
            for (const auto& p : s.client_to_server_packets)
                lst.append(nb::make_tuple(p.timestamp, nb::bytes(reinterpret_cast<const char*>(p.payload.data()), p.payload.size())));
            return lst;
        })
        .def_prop_ro("server_to_client_packets", [](const TCPSession& s) {
            nb::list lst;
            for (const auto& p : s.server_to_client_packets)
                lst.append(nb::make_tuple(p.timestamp, nb::bytes(reinterpret_cast<const char*>(p.payload.data()), p.payload.size())));
            return lst;
        })
        .def_prop_ro("messages", [](const TCPSession& s) {
            auto packet_messages = [](const std::vector<PacketRecord>& packets) {
                nb::list packet_types;
                for (const auto& packet : packets) {
                    nb::list types;
                    for (uint8_t content_type : packet.tls_record_content_types) {
                        types.append(nb::int_(content_type));
                    }
                    packet_types.append(std::move(types));
                }
                return packet_types;
            };

            nb::dict messages;
            messages["client_to_server"] = packet_messages(s.client_to_server_packets);
            messages["server_to_client"] = packet_messages(s.server_to_client_packets);
            return messages;
        });

    nb::class_<TCPSessionIterator>(m, "TCPSessionIterator")
        .def("__iter__", [](TCPSessionIterator& self) -> TCPSessionIterator& { return self; },
             nb::rv_policy::reference_internal)
        .def("__next__", [](TCPSessionIterator& it) -> TCPSession {
            auto opt = it.try_next();
            if (!opt) {
                throw nb::stop_iteration();
            }
            return std::move(*opt);
        })
        .def_prop_ro("stats",
            [](const TCPSessionIterator& it) { return it.stats(); });

    nb::class_<TCPSessionExtractor>(m, "TCPSessionExtractor")
        .def(nb::init<nb::object, nb::object, bool, std::string,
                  nb::object, nb::object, nb::object, nb::object,
                  size_t, size_t, size_t>(),
             nb::arg("pcap_path")                  = nb::none(),
             nb::arg("interface")                  = nb::none(),
             nb::arg("show_progress")              = true,
             nb::arg("bpf_filter")                 = std::string("tcp"),
             nb::arg("live_timeout")               = nb::none(),
             nb::arg("live_packet_count")          = nb::none(),
             // Hard wall-clock lifetime cap per session (seconds). Closes
             // & yields the session when exceeded. 0 or None = unlimited.
             nb::arg("max_session_duration_sec")   = nb::float_(5.0),
             // Idle/staleness cap (seconds). Open sessions older than this
             // are purged on a periodic sweep. 0 or None = disabled.
             nb::arg("stale_session_timeout_sec")  = nb::float_(30.0),
             // Per-direction storage cap (memory bound only; does NOT close
             // the session). 0 = unlimited.
             nb::arg("max_packets_per_session")    = static_cast<size_t>(0),
             // Total-packet lifecycle cap across both directions (closes &
             // yields the session when reached). 0 = unlimited.
             nb::arg("max_session_packets")        = static_cast<size_t>(0),
             // Minimum stored payload packets required in EACH direction
             // before yielding a session.
             nb::arg("min_packets_per_direction")  = static_cast<size_t>(2))
        .def("__iter__", &TCPSessionExtractor::make_iterator,
             "Return a streaming iterator.  Memory = O(active sessions).")
        .def("extract", &TCPSessionExtractor::extract,
             "Stream all sessions and return as a list.\n"
             "For large files, prefer the iterator to avoid building the full list.")
        .def_prop_ro("stats",
            [](const TCPSessionExtractor& e) {
                return ExtractorState::to_map(e.last_iter_ ? e.last_iter_->stats()
                                    : ExtractorState::statistics());
            });
}
