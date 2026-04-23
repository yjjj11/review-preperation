import redis
import threading
import time
import random


# ----------------------
# 1. 创建连接池（全局只需创建一次）
# ----------------------
max_conn = 3  # 故意设小一点，方便看争抢效果
pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=max_conn,        # 最大连接数（故意设3，方便看效果）
    socket_connect_timeout=5,
    socket_timeout=5
)

print(f"连接池初始化完成，最大连接数: {max_conn}")
print(f"初始创建连接数: {pool._created_connections}")
print()


def worker(worker_id: int, sleep_time: float):
    """模拟一个工作线程，从连接池获取连接执行操作"""
    print(f"[线程 {worker_id}] 🟡 启动，等待获取连接...")

    # 从连接池获取连接创建Redis对象
    # 这里实际是从连接池拿一个空闲连接，不是新建连接除非池子空了
    start_time = time.time()
    r = redis.Redis(connection_pool=pool)
    get_conn_time = time.time() - start_time

    with pool._lock:
        current_created = pool._created_connections
        in_use = len(pool._in_use_connections)
        available = pool.max_connections - in_use

    print(f"[线程 {worker_id}] ✅ 获取连接完成 (等待了 {get_conn_time:.3f}s)")
    print(f"[线程 {worker_id}] 📊 当前连接池状态: 已创建={current_created}, 使用中={in_use}, 可用={available}")

    # 模拟业务操作
    print(f"[线程 {worker_id}] 🔄 执行Redis操作...")
    r.set(f"worker_{worker_id}_test", f"hello_from_{worker_id}")
    value = r.get(f"worker_{worker_id}_test")

    # 模拟处理耗时
    time.sleep(sleep_time)

    print(f"[线程 {worker_id}] ✅ 完成操作，返回值: {value}")

    # 检查连接池状态
    with pool._lock:
        current_created = pool._created_connections
        in_use = len(pool._in_use_connections)
        available = pool.max_connections - in_use

    print(f"[线程 {worker_id}] 📊 操作完成后状态: 已创建={current_created}, 使用中={in_use}, 可用={available}")
    print(f"[线程 {worker_id}] 🔴 线程结束，连接归还连接池")
    print()


# ----------------------
# 2. 启动多个线程争抢连接
# ----------------------
if __name__ == "__main__":
    print("="*60)
    print(f"启动 8 个线程争抢 {max_conn} 个连接...")
    print("="*60)
    print()

    threads = []
    for i in range(8):
        # 每个线程随机休眠1-3秒，模拟不同处理时间
        sleep_t = random.uniform(1, 3)
        t = threading.Thread(target=worker, args=(i+1, sleep_t))
        threads.append(t)
        t.start()
        # 稍微错开启动，更真实
        time.sleep(0.2)

    # 等待所有线程完成
    for t in threads:
        t.join()

    # ----------------------
    # 3. 最终连接池状态
    # ----------------------
    print("="*60)
    print("所有线程执行完毕！")
    print("最终连接池状态:")
    with pool._lock:
        print(f"  - 最大连接数: {pool.max_connections}")
        print(f"  - 已创建连接: {pool._created_connections}")
        print(f"  - 使用中连接: {len(pool._in_use_connections)}")
        print(f"  - 可用连接: {pool.max_connections - len(pool._in_use_connections)}")
    print()
    print("结论:")
    print("  - 连接池会复用连接，不会一直新建连接")
    print(f"  - 即使有 8 个线程，最多只创建 {max_conn} 个连接")
    print("  - 线程用完连接会自动归还连接池给其他线程复用")

    # 清理测试数据
    r = redis.Redis(connection_pool=pool)
    for i in range(8):
        r.delete(f"worker_{i+1}_test")
    r.delete("pool_test")

    # 关闭所有连接
    pool.disconnect()
    print("\n已关闭所有连接")
