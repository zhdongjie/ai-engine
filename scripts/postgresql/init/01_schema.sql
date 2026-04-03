/*
==============================================================================
🚀 AI-Engine 核心业务表结构初始化脚本 (PostgreSQL)
==============================================================================
版本: 1.1
日期: 2026-03-26
描述: 本脚本定义了 AIOS 的核心持久化层，包括会话管理、消息流转及 pgvector 向量支持。
==============================================================================
*/

-- [0] 环境初始化
-- 开启向量存储扩展 (用于 RAG 知识库检索)
CREATE EXTENSION IF NOT EXISTS vector;


-- [1] 公共工具函数定义
-- 作用：数据库底层自动维护 updated_at 时间戳，确保数据一致性
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc', CURRENT_TIMESTAMP);
    RETURN NEW;
END;
$$ language 'plpgsql';


-- [2] 📦 核心实体：chat_sessions (会话上下文)
-- 描述：记录 AI 与用户之间的逻辑对话组
CREATE TABLE IF NOT EXISTS chat_sessions (
    -- 基础元数据
    id             UUID PRIMARY KEY,
    tenant_id      VARCHAR(36) NOT NULL,                                     -- 多租户隔离 ID
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    is_deleted     BOOLEAN DEFAULT FALSE NOT NULL,                           -- 软删除标识

    -- 业务属性
    user_id        VARCHAR(255) NOT NULL,                                    -- 关联用户 ID
    title          VARCHAR(255) DEFAULT '新对话',                             -- 会话标题
    biz_type       VARCHAR(255),                                             -- 业务类型 (如: KYC, HSBC_AUDIT)
    model_provider VARCHAR(50) DEFAULT 'openai',                             -- 模型供应商
    model_name     VARCHAR(100),                                             -- 调用的具体模型名
    system_prompt  TEXT,                                                     -- 该会话固定的系统提示词
    summary        TEXT,                                                     -- 会话自动生成的摘要
    is_pinned      BOOLEAN DEFAULT FALSE NOT NULL,                           -- 是否置顶
    lang           VARCHAR(10) DEFAULT 'ch'                                  -- 用户语言状态
);

-- 会话表索引优化
CREATE INDEX IF NOT EXISTS idx_chat_sessions_lookup ON chat_sessions(tenant_id, user_id, is_deleted);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_biz ON chat_sessions(biz_type);

-- 绑定更新触发器
CREATE TRIGGER trigger_update_chat_sessions_at
BEFORE UPDATE ON chat_sessions
FOR EACH ROW EXECUTE FUNCTION update_modified_column();


-- [3] 💬 核心实体：chat_messages (对话流水)
-- 描述：存储每一轮对话的详细内容，支持树形结构
CREATE TABLE IF NOT EXISTS chat_messages (
    -- 基础元数据
    id             UUID PRIMARY KEY,
    tenant_id      VARCHAR(36) NOT NULL,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    is_deleted     BOOLEAN DEFAULT FALSE NOT NULL,

    -- 关联属性
    -- ON DELETE CASCADE: 会话删除时自动清理相关消息
    session_id     UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id        VARCHAR(255) NOT NULL,
    
    -- 消息详情
    role           VARCHAR(20) NOT NULL,                                     -- user / assistant / system / tool
    name           VARCHAR(64),                                              -- 针对 Tool 角色的调用名
    content        TEXT NOT NULL DEFAULT '',                                 -- 文本内容
    
    -- 树形结构支持 (用于生成分支、重新生成等场景)
    parent_id      UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    position       INTEGER DEFAULT 0 NOT NULL,                               -- 消息在时间轴上的排序
    status         VARCHAR(20) DEFAULT 'completed' NOT NULL,                 -- completed / error / processing
    
    -- 扩展元数据 (JSONB 存储 Token 消耗、中间思考过程、Tool Call 信息等)
    extra          JSONB DEFAULT '{}'::jsonb NOT NULL
);

-- 消息表索引优化
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, is_deleted);
CREATE INDEX IF NOT EXISTS idx_chat_messages_tree ON chat_messages(parent_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_meta ON chat_messages USING gin(extra); -- 针对 JSONB 的全文/复杂检索

-- 绑定更新触发器
CREATE TRIGGER trigger_update_chat_messages_at
BEFORE UPDATE ON chat_messages
FOR EACH ROW EXECUTE FUNCTION update_modified_column();


-- [4] 📚 核心实体：knowledge_document_sync (知识库增量同步追踪)
-- 描述：记录本地知识库文件与向量数据库的同步状态，支持 MD5 增量比对更新
CREATE TABLE IF NOT EXISTS knowledge_document_sync (
    -- 基础元数据 (完美对齐你的 Mixin 架构)
    id             UUID PRIMARY KEY,
    tenant_id      VARCHAR(36) NOT NULL,                                     -- 多租户隔离 ID
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc') NOT NULL,
    is_deleted     BOOLEAN DEFAULT FALSE NOT NULL,                           -- 软删除标识

    -- 操作审计
    created_by     VARCHAR(255),                                             -- 记录由哪个管理员/用户触发的灌库
    updated_by     VARCHAR(255),

    -- 核心业务字段
    path_md5       VARCHAR(64) NOT NULL UNIQUE,                              -- 文件绝对路径的MD5 (唯一物理标识)
    file_path      VARCHAR(1024) NOT NULL,                                   -- 完整文件路径
    biz_type       VARCHAR(128) NOT NULL,                                    -- 业务归属 (如: java_tutor, virtual_card)
    content_hash   VARCHAR(64) NOT NULL,                                     -- 文件内容的 MD5 (用于判定是否被修改)
    chunk_count    INTEGER DEFAULT 0 NOT NULL                                -- 该文档被切分出的 chunk 数量
);

-- 知识库追踪表索引优化
-- 1. 业务高频查询：根据路径 MD5 查状态，且需排除软删除
CREATE INDEX IF NOT EXISTS idx_kb_sync_lookup ON knowledge_document_sync(path_md5, is_deleted);
-- 2. 租户与业务分类查询：用于后台管理系统统计知识库量级
CREATE INDEX IF NOT EXISTS idx_kb_sync_tenant_biz ON knowledge_document_sync(tenant_id, biz_type);
-- 3. 审计溯源：用于追踪哪些文件是谁操作的
CREATE INDEX IF NOT EXISTS idx_kb_sync_created_by ON knowledge_document_sync(created_by);

-- 绑定更新触发器 (复用你第一步定义的公共函数)
CREATE TRIGGER trigger_update_knowledge_sync_at
BEFORE UPDATE ON knowledge_document_sync
FOR EACH ROW EXECUTE FUNCTION update_modified_column();


-- [5] 额外预留：向量表说明
/* 注：langchain_pg_embedding 和 langchain_pg_collection 将由 init_knowledge_db.py 
利用 langchain_postgres 驱动在运行时动态创建。
如果需要手动预创建特定 Schema，可在此处追加 DDL。
*/