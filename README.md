

# portfolio_management

### 系統需求
Python 3.8.10^
pandas 1.4.3^
tqdm 4.64.0^
pyportfolioopt 1.5.3^
yfinance 0.1.74^
sklearn 0.0^

### 股票CSV檔格式
> 目前只會用到 Date 與 Close
> 檔名必須是 股票名.csv

| Date      | Open   | High   | Low    | Close  | Adj Close | Volume  |
| --------- | ------ | ------ | ------ | ------ | --------- | ------- |
| 2021/8/9  | 114.66 | 114.66 | 111.6  | 112.54 | 108.7597  | 1346900 |
| 2021/8/10 | 112.47 | 115.11 | 111.86 | 115.02 | 111.1563  | 1315600 |
| 2021/8/11 | 115.68 | 119.15 | 115.12 | 118.06 | 114.0942  | 2565500 |
| 2021/8/12 | 118.89 | 119.3  | 117.22 | 118.13 | 114.1619  | 1248600 |
| 2021/8/13 | 117.97 | 117.97 | 113.46 | 113.78 | 109.958   | 2293200 |
| 2021/8/16 | 113.24 | 115.91 | 112.23 | 115.36 | 111.4849  | 1833100 |
| 2021/8/17 | 113.84 | 114.09 | 109.71 | 110.75 | 107.0298  | 2212300 |
| 2021/8/18 | 110.75 | 114.1  | 110.04 | 110.1  | 106.4016  | 2079300 |

### 個股權重設定檔.json格式
```json=
{
    "BBY": {
        "upper_bound": 1.0,
        "lower_bound": -1.0
    },
    "JPM": {
        "upper_bound": 1.0,
        "lower_bound": -1.0
    },
    "MA": {
        "upper_bound": 1.0,
        "lower_bound": -1.0
    },
    "PFE": {
        "upper_bound": 0.3,
        "lower_bound": -1.0
    },
    "RRC": {
        "upper_bound": 1.0,
        "lower_bound": -1.0
    },
    "XOM": {
        "upper_bound": 1.0,
        "lower_bound": -1.0
    }
}
```

### 觀點矩陣.json格式
```json=
{
    "BBY": 0.1,
    "JPM": -0.3,
    "MA": 0.15,
    "PFE": 0.2,
    "RRC": 0.0,
    "XOM": -0.15
}
```
### 全域參數


| 指令      | 型別      | 作用     | 範例      |
| -------- | -------- | -------- | -------- |
| --mode, -M       | int(0~3) | 選擇模式 | ```-M 0 ``` |
| --input_path, -P | str | 存放股票的資料夾路徑 | ```-P ./stocksFile ``` |
| --save, -S       | boolean | 是否存配置結果，預設為否。是則儲存結果至```./result.json```，否則顯示在Terminal上 | ```-S True ``` |


## mode 0 查看單一股票資訊
#### 程式輸入範例
```python=
python finance.py --mode 0 --input_path ./stockFile --save False -SName XOM
```
#### 程式輸出結果
```python=
{'mean_pct_change': 0.002051428413795425,
 'name': 'XOM',
 'performance': {'expect_return': 0.5169599602764471,
                 'sharpe_ratio': 1.533830258593214},
 'period': {'end_time': '2022-08-09',
            'start_time': '2021-08-09',
            'stock_data_len': 253}}
```
### 參數
| 指令      | 型別      | 作用     | 範例      |
| -------- | -------- | -------- | -------- |
| --stock_name,-SName | str | 選擇單一股票，若不選擇則會回傳所有股票資訊 | ```-SName XOM ```|
## mode 1 Efficient Frontier
#### 程式輸入範例
```python=
python finance.py --mode 1 --input_path ./stockFile --save False
```
#### 程式輸出結果
```python=
{'clean_weight': OrderedDict([('BBY', 0.12143),
                              ('JPM', 0.24323),
                              ('MA', 0.1092),
                              ('PFE', 0.31461),
                              ('RRC', 0.0),
                              ('XOM', 0.21153)]),
 'performance': {'annual_volatility': 0.1914634737384096,
                 'expected_annual_return': 0.038453965238368484,
                 'sharpe_ratio': 0.09638373773360837},
 'weights': OrderedDict([('BBY', 0.1214349690896057),
                         ('JPM', 0.2432285451665681),
                         ('MA', 0.1091968330119189),
                         ('PFE', 0.3146084324987471),
                         ('RRC', 0.0),
                         ('XOM', 0.2115312202331602)])}
```
### 參數
| 指令      | 型別      | 作用     | 範例      |
| -------- | -------- | -------- | -------- |
| --weight_bounds, -WB | pair | 設定總體權重邊界，預設為(0, 1) | ```--weight_bounds -1 1 ```|
| --stocks_boundary_path, -BP | str | 個股權重設定檔，檔案格式為.json，若選擇此參數而沒有該檔案，就會在指定路徑上根據股票資訊產生一個設定檔 | ```--stocks_boundary_path ./stocks_boundary.json``` |
| --target_return, -TR | float | 輸入目標報酬，目標報酬不可大於最大報酬，不輸入則為最大報酬 |```--target_return 0.2 ```|
| --market_neutral, -MN | boolean | 設定是否採取市場中性策略，預設為否 | ```--market_neutral True ``` |
| --risk_free_rate, -RFR | float | 設定無風險資產比例，預設為0.02 | ```--risk_free_rate 0.1```|

## mode 2 Black-litterman
#### 程式輸入範例
```python=
python finance.py --mode 2 --input_path ./stockFile --save False --vews_path vews.json 
```
#### 程式輸出結果
```python=
{'clean_weight': OrderedDict([('BBY', 0.00618),
                              ('JPM', 0.0),
                              ('MA', 0.089),
                              ('PFE', 0.84326),
                              ('RRC', 0.0),
                              ('XOM', 0.06156)]),
 'performance': {'annual_volatility': 0.2756231636283055,
                 'expected_annual_return': 0.10922150844052064,
                 'sharpe_ratio': 0.3237083098024418},
 'weights': OrderedDict([('BBY', 0.0061812678636396),
                         ('JPM', 0.0),
                         ('MA', 0.0890004289081729),
                         ('PFE', 0.8432570246714299),
                         ('RRC', 0.0),
                         ('XOM', 0.0615612785567574)])}
```
### 參數
| 指令      | 型別      | 作用     | 範例      |
| -------- | -------- | -------- | -------- |
| --vews_path, -VP | str | 觀點矩陣設定檔，檔案格式為.json，若選擇此參數而沒有該檔案，就會在指定路徑上根據股票資訊產生一個設定檔 | ```--vews_path vews.json```|
| --weight_bounds, -WB | pair | 設定總體權重邊界，預設為(0, 1) | ```--weight_bounds -1 1 ```|
| --stocks_boundary_path, -BP | str | 個股權重設定檔，檔案格式為.json，若選擇此參數而沒有該檔案，就會在指定路徑上根據股票資訊產生一個設定檔 | ```--stocks_boundary_path ./stocks_boundary.json``` |
| --target_return, -TR | float | 輸入目標報酬，目標報酬不可大於最大報酬，不輸入則為最大報酬 |```--target_return 0.2 ```|
| --market_neutral, -MN | boolean | 設定是否採取市場中性策略，預設為否 | ```--market_neutral True ``` |
| --risk_free_rate, -RFR | float | 設定無風險資產比例，預設為0.02 | ```--risk_free_rate 0.1```|
## mode 3 Hierarchical Risky Party
#### 程式輸入範例
```python=
python finance.py --mode 3 --input_path ./stockFile --save False
```
#### 程式輸出結果
```python=
{'clean_weight': OrderedDict([('BBY', 0.13557),
                              ('JPM', 0.2222),
                              ('MA', 0.1471),
                              ('PFE', 0.2827),
                              ('RRC', 0.04846),
                              ('XOM', 0.16397)]),
 'performance': {'annual_volatility': 0.19918331268509046,
                 'expected_annual_return': 0.06575587587498172,
                 'sharpe_ratio': 0.2297174158726937},
 'weights': OrderedDict([('BBY', 0.1355657195749865),
                         ('JPM', 0.22219863600288395),
                         ('MA', 0.14710422332167547),
                         ('PFE', 0.2827016847255534),
                         ('RRC', 0.048455171426971894),
                         ('XOM', 0.16397456494792872)])}
```
### 參數
| 指令      | 型別      | 作用     | 範例      |
| -------- | -------- | -------- | -------- |
| --risk_free_rate, -RFR | float | 設定無風險資產比例，預設為0.02 | ```--risk_free_rate 0.1```|
| --frequency, -F | int | 一年中的交易頻率，預設為252 | ```--frequency 252```|
