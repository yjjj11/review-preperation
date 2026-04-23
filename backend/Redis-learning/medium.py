import redis
import time
import threading

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

print("="*50)
print("=== 1. 过期时间实验 ===")
print("="*50)

# 方式1：设置时指定过期时间
print("\n[实验1] 设置 temp_key 5秒过期...")
r.set("temp_key", "temp_value", ex=5)
print(f"[实验1] 立即获取: {r.get('temp_key')}")
print(f"[实验1] 当前剩余TTL: {r.ttl('temp_key')} 秒")

print("\n[实验1] 等待5秒...")
for i in range(5):
    time.sleep(1)
    ttl = r.ttl("temp_key")
    print(f"  过去 {i+1} 秒，剩余TTL: {ttl} 秒")

print(f"\n[实验1] 5秒后获取: {r.get('temp_key')}  (None表示已过期删除)")

# 方式2：事后设置过期时间
print("\n" + "-"*40)
print("\n[实验2] 先设置永久键 test_key，然后设置3秒过期...")
r.set("test_key", "test_value")
print(f"[实验2] 设置过期前TTL: {r.ttl('test_key')}  (-1表示永久)")

r.expire("test_key", 3)
print(f"[实验2] 设置过期后立即看TTL: {r.ttl('test_key')} 秒")

time.sleep(1)
print(f"[实验2] 1秒后TTL: {r.ttl('test_key')} 秒")

print(f"[实验2] 调用 persist() 取消过期...")
r.persist("test_key")
print(f"[实验2] 取消后TTL: {r.ttl('test_key')}  (-1表示永久)")

print("\n" + "="*50)
print("=== 2. Redis 发布订阅 (Pub/Sub) ===")
print("="*50)

# 回答：是的，pubsub.listen() 是**阻塞调用**
# 它会一直卡在循环里等待新消息，不会往下执行，也不会退出
# 只有当取消订阅或连接关闭时，循环才会结束

print("\n[问题解答] pubsub.listen() 是阻塞的吗？✅ 是的！")
print("它会一直阻塞等待新消息，直到订阅取消或连接关闭\n")

print("[演示] 我们用后台线程来演示，这样主线程可以继续发布消息\n")

# 定义订阅者函数（运行在后台线程）
def subscriber():
    pubsub = r.pubsub()
    pubsub.subscribe("news_channel")
    print("[订阅者] 已订阅 news_channel，开始监听... (这会阻塞这个线程)")

    count = 0
    for message in pubsub.listen():
        if message["type"] == "message":
            count += 1
            print(f"\n[订阅者] ✅ 收到消息: {message['data']}")

            # 收到3条消息后取消订阅退出阻塞
            if count >= 3:
                print("\n[订阅者] 收到3条消息，取消订阅，退出阻塞")
                pubsub.unsubscribe()
                break

# 启动订阅者线程（后台）
thread = threading.Thread(target=subscriber, daemon=True)
thread.start()

# 等待订阅者连接建立
time.sleep(0.5)

# 主线程发布消息
print("[发布者] 发布第1条消息: Hello Redis!")
r.publish("news_channel", "Hello Redis!")

time.sleep(1)
print("\n[发布者] 发布第2条消息: Pub/Sub is cool!")
r.publish("news_channel", "Pub/Sub is cool!")

time.sleep(1)
print("\n[发布者] 发布第3条消息: Bye!")
r.publish("news_channel", "Bye!")

# 等待订阅者处理完
thread.join(timeout=3)
print("\n[完成] 演示结束！要点总结：")
print("  1. ✅ listen() 确实是**阻塞调用**，会一直等消息")
print("  2.  如果不用多线程，程序会卡住不动，无法继续执行")
print("  3.  要退出阻塞，需要调用 unsubscribe() 关闭订阅")
print("  4.  Pub/Sub 是广播模式：发布一条，所有订阅者都能收到")


