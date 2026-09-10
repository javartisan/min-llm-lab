"""Python 基础：__call__，以及 *args / **kwargs 里「几个星号」是什么意思。

对照学习
    learn/week01/07_forward_logits.py 里的：
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

    tokenizer(...) 走的是 tokenizer.__call__
    model(**inputs) 里的 ** 是把字典拆成关键字参数

运行（仓库根目录）
    python learn/py/call.py

----------------------------------------------------------------------
几个星号分别是什么

    *   一个星：跟「位置参数」有关（按顺序排的，如 f(1, 2)）
    **  两个星：跟「关键字参数」有关（带名字的，如 f(x=1, y=2)）

同一个符号出现在两个地方，含义略有不同：

    写在函数【定义】上：打包（把多出来的参数收进一个变量）
        def f(*args, **kwargs):
            args    是元组，装着多出来的位置参数
            kwargs  是字典，装着多出来的关键字参数

    写在函数【调用】上：拆包（把容器拆开，当成一个个参数传进去）
        f(*[1, 2])           等价于 f(1, 2)
        f(**{"x": 1, "y": 2}) 等价于 f(x=1, y=2)

记一句：定义时打包，调用时拆开；一星按顺序，两星按名字。
"""


class Call:
    """演示「对象可以当函数用」：实例后面加括号，会进 __call__。"""

    def __call__(self, *args, **kwargs):
        # *args    ：多出来的位置参数，收成元组，例如 (1, 2)
        # **kwargs ：多出来的关键字参数，收成字典，例如 {"name": "Java"}
        print("  __call__ 收到 args   =", args, "  类型", type(args).__name__)
        print("  __call__ 收到 kwargs =", kwargs, "  类型", type(kwargs).__name__)
        return str(self) + " => called"


def demo_define_star():
    """定义时的 * / ** ：打包多出来的参数。"""

    def pack(a, *args, **kwargs):
        print("  普通参数 a =", a)
        print("  *args 打包 =", args)
        print("  **kwargs 打包 =", kwargs)

    print("=" * 60)
    print("【1】定义里的 *args / **kwargs = 打包")
    print("=" * 60)
    print("调用 pack(10, 20, 30, x=1, y=2)")
    pack(10, 20, 30, x=1, y=2)
    print("读法：10 给 a；20、30 进 args 元组；x、y 进 kwargs 字典。")


def demo_call_star():
    """调用时的 * / ** ：把列表/字典拆开再传进去。"""

    def add(x, y):
        return x + y

    print()
    print("=" * 60)
    print("【2】调用里的 * / ** = 拆包")
    print("=" * 60)
    nums = [3, 4]
    print("add(*nums) 其中 nums =", nums, " →", add(*nums), "  等价 add(3, 4)")

    kv = {"x": 10, "y": 20}
    print("add(**kv) 其中 kv =", kv, " →", add(**kv), "  等价 add(x=10, y=20)")
    print("注意：字典的键必须和参数名一致，否则会报 unexpected keyword argument。")


def demo_model_style():
    """和 07 脚本里 model(**inputs) 同一种写法。"""

    def fake_model(input_ids, attention_mask):
        print("  fake_model 收到 input_ids      =", input_ids)
        print("  fake_model 收到 attention_mask =", attention_mask)
        return "logits 占位"

    print()
    print("=" * 60)
    print("【3】对照 07：model(**inputs)")
    print("=" * 60)
    inputs = {
        "input_ids": [39570, 8334],
        "attention_mask": [1, 1],
    }
    print("inputs 是字典:", inputs)
    print("fake_model(**inputs) 拆成两个带名字的参数：")
    result = fake_model(**inputs)
    print("返回:", result)
    print("所以 ** 的个数=2，表示：按「名字」把字典拆进参数列表。")


def demo_call_object():
    print()
    print("=" * 60)
    print("【4】对象 + 括号 = 调用 __call__")
    print("=" * 60)
    call = Call()
    print("call() 无参数：")
    print("  返回:", call())
    print("call(1, 2, name='Java')：")
    print("  返回:", call(1, 2, name="Java"))


def main():
    print("星号个数：* 管位置参数，** 管关键字参数。")
    print("出现位置：定义=打包，调用=拆开。")
    demo_define_star()
    demo_call_star()
    demo_model_style()
    demo_call_object()
    print()
    print("小结论：")
    print("  - *args    一个星，元组，按顺序多出来的参数")
    print("  - **kwargs 两个星，字典，按名字多出来的参数")
    print("  - 调用时 *列表 / **字典 是反过来：拆开再传入")
    print("  - model(**inputs) 就是把 input_ids、attention_mask 按名字传给模型")


if __name__ == "__main__":
    main()
