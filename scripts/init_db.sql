-- ==============================================================================
-- 🚀 AI-Engine 核心业务表结构初始化脚本 (PostgreSQL)
-- ==============================================================================

-- 1. 创建自动更新 updated_at 的触发器函数
-- 作用：任何时候只要执行了 UPDATE 操作，数据库底层自动更新时间，不用依赖 Python 侧
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc', CURRENT_TIMESTAMP);
    RETURN NEW;
END;
$$ language 'plpgsql';


-- ==========================================
-- 📦 1. 表: chat_sessions (会话主表)
-- ==========================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    -- [Base & Mixin 字段]
    id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', CURRENT_TIMESTAMP) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', CURRENT_TIMESTAMP) NOT NULL,
    tenant_id VARCHAR(36) NOT NULL,
    is_deleted BOOLEAN DEFAULT FALSE NOT NULL,

    -- [业务字段]
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) DEFAULT '新对话',
    biz_type VARCHAR(255),
    model_provider VARCHAR(50) DEFAULT 'openai',
    model_name VARCHAR(100),
    system_prompt TEXT,
    summary TEXT,
    is_pinned BOOLEAN DEFAULT FALSE NOT NULL
);

-- 为高频查询字段创建索引
CREATE INDEX idx_chat_sessions_tenant_id ON chat_sessions(tenant_id);
CREATE INDEX idx_chat_sessions_is_deleted ON chat_sessions(is_deleted);
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_biz_type ON chat_sessions(biz_type);

-- 绑定更新时间触发器
CREATE TRIGGER trigger_update_chat_sessions_updated_at
BEFORE UPDATE ON chat_sessions
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();


-- ==========================================
-- 💬 2. 表: chat_messages (消息流水表)
-- ==========================================
CREATE TABLE IF NOT EXISTS chat_messages (
    -- [Base & Mixin 字段]
    id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', CURRENT_TIMESTAMP) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', CURRENT_TIMESTAMP) NOT NULL,
    tenant_id VARCHAR(36) NOT NULL,
    is_deleted BOOLEAN DEFAULT FALSE NOT NULL,

    -- [业务字段]
    -- ON DELETE CASCADE: 当会话被物理删除时，底下的消息自动级联删除
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    name VARCHAR(64),
    content TEXT NOT NULL DEFAULT '',

    -- parent_id 用于支持树形对话(如重新生成)，自引用外键
    parent_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    position INTEGER DEFAULT 0 NOT NULL,
    status VARCHAR(20) DEFAULT 'completed' NOT NULL,

    -- JSONB 格式的高级元数据存储
    extra JSONB DEFAULT '{}'::jsonb NOT NULL
);

-- 为高频查询字段创建索引
CREATE INDEX idx_chat_messages_tenant_id ON chat_messages(tenant_id);
CREATE INDEX idx_chat_messages_is_deleted ON chat_messages(is_deleted);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id);
CREATE INDEX idx_chat_messages_parent_id ON chat_messages(parent_id);
CREATE INDEX idx_chat_messages_position ON chat_messages(position);

-- 绑定更新时间触发器
CREATE TRIGGER trigger_update_chat_messages_updated_at
BEFORE UPDATE ON chat_messages
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- ==============================================================================
-- ✅ 初始化完成
-- ==============================================================================