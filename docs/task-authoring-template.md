# Direct-PTX 任务模板入口

新建和发布任务只使用 Docker-only `direct-ptx-v1`：

- [`../tasks/README.md`](../tasks/README.md)：目录、生成、验证和运行流程；
- [`../tasks/DESIGN.md`](../tasks/DESIGN.md)：选题、正确性、性能和校准原则；
- [`../templates/direct-ptx-task-v1/`](../templates/direct-ptx-task-v1/)：唯一受支持模板。

旧 v0.x pilot 只保留运行证据，不再提供模板或 checker fallback。需要继续使用的题目
必须用 v1 scaffold 重新生成并逐题迁移。
