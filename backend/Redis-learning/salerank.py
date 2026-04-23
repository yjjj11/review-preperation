import redis

class ProductRank:
    def __init__(self):
        self.r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self.rank_key = "product:sales:rank"

    def add_sales(self, product_id, sales):
        """增加商品销量"""
        self.r.zincreby(self.rank_key,sales,product_id)     

    def get_top_n(self,n):
        "获取销量排名前n的商品列表]"
        return self.r.zrevrange(self.rank_key,0,n-1,withscores=True)
    
    def get_sales(self ,product_id):
        "获取单个商品的销量"
        return self.r.zscore(self.rank_key,product_id)

# 测试代码
if __name__ == "__main__":
    rank = ProductRank()

    # 模拟销量数据
    rank.add_sales("product_001", 100)
    rank.add_sales("product_002", 250)
    rank.add_sales("product_003", 180)
    rank.add_sales("product_004", 300)

    # 获取销量前三
    print("销量前三商品：")
    top3 = rank.get_top_n(3)
    for idx, (pid, sales) in enumerate(top3, 1):
        print(f"第{idx}名: {pid} → 销量 {int(sales)}")

    # 获取单个商品销量
    print(f"\nproduct_002 销量: {rank.get_sales('product_002')}")