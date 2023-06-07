# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import streamlit as st
import random
import statistics

st.title("随机数统计")

n = st.number_input("请输入数字n", min_value=1, step=1)
random_nums = [random.randint(1, 100) for _ in range(n)]

st.write("随机生成的数字为：", random_nums)

if len(random_nums) < 2:
    st.write("输入的数字数量太少，无法计算方差")
else:
    mean = statistics.mean(random_nums)
    variance = statistics.variance(random_nums)
    st.write(f"这{n}个随机数字的平均值为：{mean:.2f}")
    st.write(f"这{n}个随机数字的方差为：{variance:.2f}")


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
