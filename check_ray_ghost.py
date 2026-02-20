import ray
from ray.util.placement_group import get_placement_group

# 尝试连接现有的 Ray 集群
try:
    ray.init(address="auto", ignore_reinit_error=True)
    print("✅ 成功连接到 Ray 集群！")
except Exception as e:
    print(f"❌ 无法连接到 Ray (可能已经死透了): {e}")
    exit()

# 检查报错里的那个具体名字
target_name = "global_poolverl_group_2:0"

try:
    # 尝试获取这个 group
    pg = get_placement_group(target_name)
    if pg:
        print(f"\n👻 抓到了！发现僵尸 Placement Group: {target_name}")
        print(f"   状态: {pg.state}")
        print(f"   ID: {pg.id}")
    else:
        print(f"\n🤷‍♂️ 没找到名为 {target_name} 的 Group，可能已经清理了。")
except ValueError:
    print(f"\n🤷‍♂️ 没找到名为 {target_name} 的 Group (ValueError)。")
except Exception as e:
    print(f"查询出错: {e}")

# 列出所有存在的 Groups
print("\n📋 当前所有存在的 Placement Groups:")
from ray.util.state import list_placement_groups

try:
    pgs = list_placement_groups()
    for item in pgs:
        print(f"- Name: {item.get('name', 'NoName')} | State: {item.get('state')}")
except:
    print("无法列出详细列表")
