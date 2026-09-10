# Instruct 底座 vs 微调模型 对比测试报告

- 生成时间: `2026-09-09 18:14:18`
- Instruct 底座: `/Users/tengdelong/PycharmProjects/SmolLM2-135M-sft/models/SmolLM2-135M-Instruct`
- 微调模型: `/Users/tengdelong/PycharmProjects/SmolLM2-135M-sft/models/smollm2-135m-java-sft`
- 测试集: `/Users/tengdelong/PycharmProjects/SmolLM2-135M-sft/data/test.jsonl`（12 条）
- 生成参数: `max_new_tokens=128`, `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.1`

## 指标说明

| 指标 | 含义 | 方向 |
|---|---|---|
| NLL | 参考答案在模型下的平均负对数似然 | 越低越好 |
| Token F1 | 生成文本与参考答案的字符集合 F1 | 越高越好 |
| Keyword Hit | 参考答案关键片段在生成中的命中率 | 越高越好 |
| Latency | 单条生成耗时（秒） | 越低越好 |

## 汇总对比

| 指标 | Instruct 底座 | 微调模型 | 更优方 |
|---|---:|---:|---|
| NLL | 2.4676 | 1.5513 | 微调模型 |
| Token F1 | 0.2360 | 0.2095 | Instruct 底座 |
| Keyword Hit | 0.1311 | 0.0792 | Instruct 底座 |
| Latency (s) | 2.570 | 2.332 | - |

## 逐条生成对比

### 用例 1

**Prompt:** 请解释一下 Java 里的 interface 是什么？

**Reference:** interface 用于声明一组方法规范，供类去实现。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | Java 的 interface 是以 `interface` 类型定义为基本实现类型, 这个特性在 Java 5.0 之后提到: | 2.8818 | 0.3750 | 0.1429 | 4.431 |
| 微调 | Java 1.6特性后,接口就会自动生成一个实现类型的对象。 | 1.8150 | 0.2545 | 0.0714 | 2.471 |

### 用例 2

**Prompt:** Java 接口能不能用 new 创建对象？

**Reference:** 不能，interface 不能直接被实例化。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | No, you cannot use `new` to create a new object in Java. You need to explicitly declare the class as an instance of a different class or interface. Here's how: | 2.9359 | 0.3600 | 0.1250 | 1.598 |
| 微调 | 可以创建垃圾对象、异常对象和非默认类型对象。 | 1.7185 | 0.0541 | 0.0000 | 1.827 |

### 用例 3

**Prompt:** 继承在 Java 中有什么作用？

**Reference:** 子类可以通过继承复用父类已有的属性和方法。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | In Java, inheritance is a mechanism that allows you to create new classes based on existing ones. It's used for both object-oriented and functional programming purposes. Here are some common scenarios where inheritance is used: | 2.1594 | 0.0000 | 0.0000 | 2.181 |
| 微调 | 可适配、多态、冗弄、重载、约束、实现类型、方法参数和接口属于哪个类？ | 1.2859 | 0.2500 | 0.0526 | 2.450 |

### 用例 4

**Prompt:** 请简单说明 Java 多态的含义。

**Reference:** 多态指同一行为在不同对象上可以有不同实现。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | Java 的多态（multiple-inheritance）和类型推理（type coercion）是两个重点的技术来解决闭包、基本类型转换和异常判断的工作原因： | 2.4701 | 0.0556 | 0.0556 | 4.454 |
| 微调 | 多态：可以实现统一行内存分配、线程安全、异步执行；小型和大型类型有别的作用；构造方法、静态方法、static成员、final化方法。 | 1.4652 | 0.2941 | 0.2222 | 3.755 |

### 用例 5

**Prompt:** abstract class 和 interface 有什么区别？

**Reference:** abstract class 可以有实现代码，interface 更侧重定义规范。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | Abstract Class and Interface 的区别在于： | 2.6160 | 0.4490 | 0.2000 | 0.668 |
| 微调 | Abstract类必须实现接口，定义了多个方法，可以在同一个继承子类上加入依赖。 | 1.8533 | 0.4375 | 0.2667 | 2.046 |

### 用例 6

**Prompt:** 一个类可以实现多个 interface 吗？

**Reference:** 可以，Java 支持一个类实现多个接口。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在Java 中，一个类和一个接受了一个或者多个接受到的对象时, 都能够使用同样的 interface 来定义一个接受过来的对象. | 1.8842 | 0.3333 | 0.4167 | 3.101 |
| 微调 | 如果接口定义了多个子类，那就会导致继承引用计数过于大，因此通常应该使用构造方法来创建子类。 | 1.0240 | 0.2373 | 0.1667 | 3.430 |

### 用例 7

**Prompt:** Java 中 extends 关键字用来做什么？

**Reference:** extends 用于声明类继承另一个类。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在 Java 中，`extends` 是一个关键字，它可以包含多个关键字或者一个名称为 `subclass` 的类型。 | 3.1095 | 0.3860 | 0.1818 | 2.532 |
| 微调 | 包含所有子类和接口的实例化方法。 | 1.6204 | 0.1176 | 0.0000 | 0.907 |

### 用例 8

**Prompt:** 方法重写 override 和多态有什么关系？

**Reference:** 子类重写父类方法后，运行时会按实际对象类型调用对应实现，体现多态。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在这个问题中，我将使用override来触发动区和多态去继承类型而不是抽取了新类型。 | 2.2901 | 0.2154 | 0.0741 | 2.318 |
| 微调 | override 是子类定义的特化，而覆盖就是子类定义的特化。 | 1.7565 | 0.1667 | 0.0370 | 1.448 |

### 用例 9

**Prompt:** interface 里的方法默认是什么访问权限？

**Reference:** 接口中的方法默认是 public abstract。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在这个示例中，我将使用了一个接受参数的方法和接受过程的字符串类型作为参数的接受方式。 | 2.2715 | 0.2264 | 0.2000 | 2.325 |
| 微调 | 可选、自定义、非常明执行时刻保护异常和多线程安全性。 | 1.5714 | 0.0444 | 0.0000 | 1.966 |

### 用例 10

**Prompt:** 父类引用可以指向子类对象吗？

**Reference:** 可以，这是多态中向上转型的常见写法。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在 Java 8 中，如果我们使用了 `this` 指针来访问父类的属性和方法，则可以通过 `this` 指针来相互转换为子类对象。 | 2.8615 | 0.2462 | 0.0714 | 3.587 |
| 微调 | 可能有问题：如果子类重写了父类方法，那么将会覆盖父类实例。 | 2.0493 | 0.2273 | 0.0000 | 2.260 |

### 用例 11

**Prompt:** 为什么说 interface 适合做解耦？

**Reference:** 调用方只依赖接口约定，不依赖具体实现类。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在Java 里, interface is a powerful concept that helps developers write more concise and maintainable code. Here's why: | 2.2626 | 0.0000 | 0.0000 | 1.234 |
| 微调 | 默认接口可以实现所有类型，固定接口则必须重写所有接口方法，避免多态、单例式异常和非线程安全问题。 | 1.3504 | 0.3051 | 0.1333 | 3.149 |

### 用例 12

**Prompt:** Java 单继承限制指的是什么？

**Reference:** 一个类只能直接继承一个父类，但可以实现多个接口。

| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |
|---|---|---:|---:|---:|---:|
| Instruct 底座 | 在 Java 中, `java.lang.Object` 是一个类型为对象的数值类型，而 `java.util.Objects` 是一个类型为对象的数值类型。 | 1.8686 | 0.1852 | 0.1053 | 2.413 |
| 微调 | 单线程、自定义类型、多态、垃圾链表、面向对象、异常捕获和重载方法。 | 1.1060 | 0.1250 | 0.0000 | 2.279 |

## 结论

- 在参考答案似然（NLL）上，更优: **微调模型**（底座 2.4676 vs 微调 1.5513）。
- 在字符级重叠（Token F1）上，更优: **Instruct 底座**（底座 0.2360 vs 微调 0.2095）。
- 在关键词命中上，更优: **Instruct 底座**（底座 0.1311 vs 微调 0.0792）。
- 本测试使用独立测试集（含改写题与相关新题），未使用训练数据；样本量仍较小，结论仅供参考。
