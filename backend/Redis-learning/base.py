import redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
 
r.set("name", "redis")
r.set("count", 1)
# 设置过期时间（10秒后自动删除）
r.set("token", "abc123", ex=10)

# 2. 获取值
print(r.get("name"))  # 输出 redis
print(r.get("count"))  # 输出 1

# 3. 数字自增/自减（计数器核心）
r.incr("count")  # count 变为 2
r.incr("count", 2)  # 步长2，count 变为 4
r.decr("count")  # count 变为 3

# 4. 批量设置/获取
r.mset({"k1": "v1", "k2": "v2", "k3": "v3"})
print(r.mget(["k1", "k2", "k3"]))  # 输出 ['v1', 'v2', 'v3']

# 5. 查看过期时间（-1=永久，-2=已过期）
print(r.ttl("token"))  # 输出剩余秒数，10秒后变为-2


r.hset("user:1", "name", "张三")
r.hset("user:1", "age", 20)

# 2. 批量存储字段
r.hset("user:1", mapping={"gender": "男", "city": "杭州"})

# 3. 获取单个/所有字段
print(r.hget("user:1", "name"))  # 输出 张三
print(r.hgetall("user:1"))  # 输出 {'name':'张三','age':'20','gender':'男','city':'杭州'}

# 4. 获取所有字段名/值
print(r.hkeys("user:1"))  # 输出 ['name','age','gender','city']
print(r.hvals("user:1"))  # 输出 ['张三','20','男','杭州']

# 5. 删除字段
r.hdel("user:1", "city")



# 1. 左插/右插元素
r.lpush("mylist", "a", "b", "c")  # 左边插入，列表变为 [c,b,a]
r.rpush("mylist", "d", "e")       # 右边插入，列表变为 [c,b,a,d,e]

# 2. 获取列表元素（0=第一个，-1=最后一个）
print(r.lrange("mylist", 0, -1))  # 输出 ['c','b','a','d','e']

# 3. 左弹/右弹元素（弹出即删除）
print(r.lpop("mylist"))  # 弹出最左边元素 c
print(r.rpop("mylist"))  # 弹出最右边元素 e
print(r.lrange("mylist", 0, -1))  # 剩余 ['b','a','d']

# 4. 获取列表长度
print(r.llen("mylist"))  # 输出 3


# 1. 添加元素（自动去重）
r.sadd("myset", 1, 2, 2, 3)  # 实际存储 {1,2,3}

# 2. 获取所有元素
print(r.smembers("myset"))  # 输出 {'1','2','3'}（无序）

# 3. 判断元素是否存在
print(r.sismember("myset", 2))  # 输出 True
print(r.sismember("myset", 4))  # 输出 False

# 4. 集合运算（交集/并集/差集）
r.sadd("set1", "a", "b", "c")
r.sadd("set2", "b", "c", "d")
print(r.sinter("set1", "set2"))  # 交集 {'b','c'}
print(r.sunion("set1", "set2"))  # 并集 {'a','b','c','d'}
print(r.sdiff("set1", "set2"))   # 差集 {'a'}

# 5. 删除元素
r.srem("myset", 3)


# 1. 添加元素（score: member）
r.zadd("rank", {"小明": 100, "小红": 200, "小刚": 150})

# 2. 按分数升序/降序获取
print(r.zrange("rank", 0, -1))  # 升序 ['小明','小刚','小红']
# 带分数输出
print(r.zrange("rank", 0, -1, withscores=True))  # [('小明',100.0),('小刚',150.0),('小红',200.0)]

# 3. 降序获取（排行榜常用）
print(r.zrevrange("rank", 0, -1, withscores=True))  # [('小红',200.0),('小刚',150.0),('小明',100.0)]

# 4. 分数增减（更新排名）
r.zincrby("rank", 50, "小明")  # 小明分数+50 → 150
print(r.zrevrange("rank", 0, -1, withscores=True))  # 小明和小刚并列第二

# 5. 获取前三名
print(r.zrevrange("rank", 0, 2, withscores=True))


print(r.exists("name"))  # 存在返回1，不存在返回0

# 2. 查看所有键（生产环境慎用 KEYS *）
print(r.keys())  # 输出所有键名列表

# 3. 删除键
r.delete("k1", "k2")  # 可批量删除

# 4. 查看键的类型
print(r.type("user:1"))  # 输出 hash
print(r.type("mylist"))  # 输出 list