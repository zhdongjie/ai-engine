# 虚拟卡开发者接口文档 (v2.1)

## 核心接口列表

### 1. 实时冻结接口
* **Endpoint**: `/api/v2/card/freeze`
* **Method**: `POST`
* **Params**: `card_id`, `reason`
* **描述**: 用于用户在 App 中点击“一键锁卡”时调用。

### 2. 消费额度调整
* **Endpoint**: `/api/v2/card/limit-adjust`
* **Method**: `PUT`
* **描述**: 调整每日限额。单次调增不得超过原额度的 20%。

### 3. 状态查询
* **状态码说明**:
  - `0`: 正常 (Active)
  - `1`: 已锁定 (Locked)
  - `2`: 已注销 (Terminated)