import polars as pl


__all__ = ['packets_to_netflow']


def packets_to_netflow(packets_data, HOST_IP):
    # Convert the packet data into a Polars DataFrame
    df = pl.DataFrame(packets_data)

    # Convert TIMESTAMP to milliseconds since epoch for easier calculation
    df = df.with_columns([
        (pl.col("TIMESTAMP").cast(pl.Datetime).dt.timestamp().cast(
            pl.Float64) * 1000).alias("EPOCH_MILLISECONDS")
    ]).drop("TIMESTAMP")

    # Perform aggregation to calculate start, end times, and total bytes per flow
    aggregated_df = df.group_by(["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO"]).agg([
        pl.sum("IN_BYTES").alias("IN_BYTES"),
        pl.sum("IN_PKTS").alias("IN_PKTS"),
        pl.sum("OUT_BYTES").alias("OUT_BYTES"),
        pl.sum("OUT_PKTS").alias("OUT_PKTS"),
        pl.min("EPOCH_MILLISECONDS").alias("FLOW_START_MS"),
        pl.max("EPOCH_MILLISECONDS").alias("FLOW_END_MS"),
        pl.sum("PACKET_LENGTH").alias("TOTAL_BYTES"),
        pl.first("TCP_FLAGS").alias("TCP_FLAGS"),
        pl.first("CLIENT_TCP_FLAGS").alias("CLIENT_TCP_FLAGS"),
        pl.first("SERVER_TCP_FLAGS").alias("SERVER_TCP_FLAGS"),
        pl.min("MIN_TTL").alias("MIN_TTL"),
        pl.max("MAX_TTL").alias("MAX_TTL"),
        pl.max("LONGEST_FLOW_PKT").alias("LONGEST_FLOW_PKT"),
        pl.min("SHORTEST_FLOW_PKT").alias("SHORTEST_FLOW_PKT"),
        pl.min("MIN_IP_PKT_LEN").alias("MIN_IP_PKT_LEN"),
        pl.max("MAX_IP_PKT_LEN").alias("MAX_IP_PKT_LEN"),
        pl.sum("RETRANSMITTED_IN_BYTES").alias("RETRANSMITTED_IN_BYTES"),
        pl.sum("RETRANSMITTED_IN_PKTS").alias("RETRANSMITTED_IN_PKTS"),
        pl.sum("RETRANSMITTED_OUT_BYTES").alias("RETRANSMITTED_OUT_BYTES"),
        pl.sum("RETRANSMITTED_OUT_PKTS").alias("RETRANSMITTED_OUT_PKTS"),
        pl.mean("SRC_TO_DST_AVG_THROUGHPUT").alias("SRC_TO_DST_AVG_THROUGHPUT"),
        pl.mean("DST_TO_SRC_AVG_THROUGHPUT").alias("DST_TO_SRC_AVG_THROUGHPUT"),
        pl.sum("NUM_PKTS_UP_TO_128_BYTES").alias("NUM_PKTS_UP_TO_128_BYTES"),
        pl.sum("NUM_PKTS_128_TO_256_BYTES").alias("NUM_PKTS_128_TO_256_BYTES"),
        pl.sum("NUM_PKTS_256_TO_512_BYTES").alias("NUM_PKTS_256_TO_512_BYTES"),
        pl.sum("NUM_PKTS_512_TO_1024_BYTES").alias("NUM_PKTS_512_TO_1024_BYTES"),
        pl.sum("NUM_PKTS_1024_TO_1514_BYTES").alias("NUM_PKTS_1024_TO_1514_BYTES"),
        pl.max("TCP_WIN_MAX_IN").alias("TCP_WIN_MAX_IN"),
        pl.max("TCP_WIN_MAX_OUT").alias("TCP_WIN_MAX_OUT"),
        pl.max("ICMP_TYPE").alias("ICMP_TYPE"),
        pl.max("ICMP_IPV4_TYPE").alias("ICMP_IPV4_TYPE"),
        pl.max("DNS_QUERY_ID").alias("DNS_QUERY_ID"),
        pl.max("DNS_QUERY_TYPE").alias("DNS_QUERY_TYPE"),
        pl.max("DNS_TTL_ANSWER").alias("DNS_TTL_ANSWER"),
        pl.max("FTP_COMMAND_RET_CODE").alias("FTP_COMMAND_RET_CODE")
    ])

    # First, calculate flow duration in milliseconds
    aggregated_df = aggregated_df.with_columns([
        (pl.col("FLOW_END_MS") - pl.col("FLOW_START_MS")).alias("FLOW_DURATION_MILLISECONDS")]
    )

    # Calculate DURATION_IN and DURATION_OUT
    aggregated_df = aggregated_df.with_columns([
        (pl.col("FLOW_DURATION_MILLISECONDS") / 1000 * pl.when(pl.col("IPV4_SRC_ADDR")
         == HOST_IP).then(1).otherwise(0)).cast(pl.Int64).alias("DURATION_IN"),
        (pl.col("FLOW_DURATION_MILLISECONDS") / 1000 * pl.when(pl.col("IPV4_DST_ADDR")
         == HOST_IP).then(1).otherwise(0)).cast(pl.Int64).alias("DURATION_OUT"),
    ])

    # Then, calculate bytes per second based on the flow duration
    aggregated_df = aggregated_df.with_columns([
        pl.when(pl.col("FLOW_DURATION_MILLISECONDS") > 0)
        .then(pl.col("TOTAL_BYTES") / (pl.col("FLOW_DURATION_MILLISECONDS") / 1000))
        .otherwise(0).alias("BYTES_PER_SECOND")]
    ).drop(["FLOW_START_MS", "FLOW_END_MS", "TOTAL_BYTES"])

    # Calculate SRC_TO_DST_AVG_THROUGHPUT and DST_TO_SRC_AVG_THROUGHPUT
    aggregated_df = aggregated_df.with_columns([
        ((pl.col("IN_BYTES") / pl.when(pl.col("DURATION_IN") > 0).then(pl.col("DURATION_IN")).otherwise(1)) *
         pl.when(pl.col("FLOW_DURATION_MILLISECONDS") > 0).then(pl.col("FLOW_DURATION_MILLISECONDS")).otherwise(1)).cast(pl.Int64).alias("SRC_TO_DST_AVG_THROUGHPUT"),

        ((pl.col("OUT_BYTES") / pl.when(pl.col("DURATION_OUT") > 0).then(pl.col("DURATION_OUT")).otherwise(1)) *
         pl.when(pl.col("FLOW_DURATION_MILLISECONDS") > 0).then(pl.col("FLOW_DURATION_MILLISECONDS")).otherwise(1)).cast(pl.Int64).alias("DST_TO_SRC_AVG_THROUGHPUT"),
    ])

    # Calculate SRC_TO_DST_AVG_THROUGHPUT and DST_TO_SRC_AVG_THROUGHPUT
    aggregated_df = aggregated_df.with_columns([
        (pl.sum("IN_BYTES") * pl.when(pl.col("BYTES_PER_SECOND") > 0).then(pl.col("BYTES_PER_SECOND")
                                                                           ).otherwise(1)).cast(pl.Int64).alias("DST_TO_SRC_SECOND_BYTES"),
        (pl.sum("OUT_BYTES") * pl.when(pl.col("BYTES_PER_SECOND") > 0).then(pl.col("BYTES_PER_SECOND")
                                                                            ).otherwise(1)).cast(pl.Int64).alias("SRC_TO_DST_SECOND_BYTES"),
    ]).drop(["BYTES_PER_SECOND"])

    return aggregated_df
