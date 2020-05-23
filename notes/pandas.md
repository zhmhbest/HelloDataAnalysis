<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Pandas](./index.html)


```py
    # 生成时间序列
    # start end : 时间起止
    # periods : 生成个数
    # freq : 生成频率
    """
    D : Day
    B : BusinessDay
    H : Hour
    T/min : Minutes
    S : Second
    L/ms : Milli
    U : Micro
    M : MonthEnd
    BM : BusinessMonthEnd
    MS : MonthBegin
    BMS : BusinessMonthBegin
    """
    print('时间序列')
    df = pd.date_range("1996-10-16", "2019-10-25", periods=None, freq='Y')
    print(len(df))
    print(df[0])
    print(df)
```