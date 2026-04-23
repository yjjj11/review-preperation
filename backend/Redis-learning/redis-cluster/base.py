import redis
from redis.cluster import RedisCluster, ClusterConnectionPool

def get_cluster_client():
    """生产环境：获取集群客户端（带连接池）"""
    pool = ClusterConnectionPool(
        host="localhost",
        port=7000,
        decode_responses=True,
        max_connections_per_node=20,
        max_connections=100,
        cluster_error_retry_attempts=3,
        retry_on_timeout=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    return RedisCluster(connection_pool=pool)

def main():
    r = get_cluster_client()

    # 1. 基本操作
    print("=== 基本操作 ===")
    r.set("name", "redis_cluster")
    print(f"get name: {r.get('name')}")

    # 2. Hash Tag 多键操作
    print("\n=== Hash Tag 多键操作 ===")
    user_tag = "{user:1001}"
    r.mset({
        f"{user_tag}:order": "order_123",
        f"{user_tag}:cart": "cart_456"
    })
    print(f"mget: {r.mget(f'{user_tag}:order', f'{user_tag}:cart')}")

    # 3. 集群事务（同槽 key）
    print("\n=== 集群事务 ===")
    tx_tag = "{tx}"
    pipe = r.pipeline(transaction=True)
    pipe.set(f"{tx_tag}:k1", "v1")
    pipe.set(f"{tx_tag}:k2", "v2")
    pipe.get(f"{tx_tag}:k1")
    print(f"事务结果: {pipe.execute()}")

    # 4. 集群状态检查
    print("\n=== 集群状态 ===")
    cluster_info = r.cluster_info()
    print(f"集群状态: {cluster_info['cluster_state']}")
    print(f"节点数: {cluster_info['cluster_known_nodes']}")

if __name__ == "__main__":
    main()