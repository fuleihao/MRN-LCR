# RelationNegotiator 注意力聚合改进说明

本文档说明将 `RelationNegotiator` 的客体→主体聚合从“简单均值”替换为“注意力加权”的改进思路、原因与实现要点。

## 背景问题
- 原实现对客体聚合为：对参与对话的所有客体状态做均值 `(mean)`，再反馈给主体。
- 在关系抽取中，候选客体数量通常多且负样本居多，均值会显著稀释少数正样本的强信号，导致主体更新方向偏向“全局背景”。
- 多轮对话会重复注入均值噪声，可能出现过平滑（oversmoothing）。

## 改进目标
- 让主体更关注“最相关”的客体，把强相关客体的信号以更高权重传回主体，降低弱相关或负样本的影响。

## 核心思路
1. 引入轻量的注意力打分模块 `attn_scorer(MLP)`，对每个主体-客体对输出标量相关性：
   - 输入特征：`concat(object_state, subject_state)`。
   - 打分后对掩码位置施加 `-inf`（实现为一个大负数），并用 `softmax` 归一化为权重。
2. 用注意力权重对客体状态加权求和，得到聚合向量：`aggregated = sum(att_i * object_state_i)`。
3. 保留稳健性：当掩码全零或数值不稳时，回退为原均值聚合，避免异常。

## 接口与兼容性
- 改动仅在 `RelationNegotiator.forward` 内部实现，保持接口不变：
  - 输入：`subject_states (bsz, H)`, `object_states (bsz, ent_len, H)`, `pair_mask (bsz, ent_len)`。
  - 输出：更新后的 `subject_states` 与 `object_states`。
- 多轮机制 `num_dialogue_rounds` 保持原样；每轮使用注意力聚合替代均值。

## 与候选增强的协同
- 当前候选增强 `_candidate_enrichment` 在分类后进行，用得分选择若干关系标签嵌入增强客体特征。
- 注意力聚合独立于候选增强，但二者可互补：候选增强提升客体表示的辨识度，注意力聚合把更可靠的客体信号反馈给主体。
- 若需要更紧耦合，可将候选得分或 NER 类型兼容性注入注意力打分（后续可扩展）。

## 代码要点（摘要）
- 新增成员：`self.attn_scorer = nn.Sequential(Linear(2H->H), GELU, Linear(H->1))`。
- 客体→主体阶段：
  ```python
  subj_expanded = subject_states.unsqueeze(1).expand_as(object_states)
  pair_features = torch.cat([object_states, subj_expanded], dim=-1)
  attn_logits = self.attn_scorer(pair_features).squeeze(-1)
  attn_logits = attn_logits.masked_fill(~pair_mask.bool(), -1e4)
  attn = torch.softmax(attn_logits, dim=-1)
  aggregated_attn = torch.sum(object_states * attn.unsqueeze(-1), dim=1)
  # 回退：当有效客体为空或数值异常，使用均值
  denom = pair_mask.to(dtype=object_states.dtype).unsqueeze(-1).sum(dim=1).clamp_min(self.eps)
  aggregated_mean = torch.sum(object_states * pair_mask.unsqueeze(-1).to(object_states.dtype), dim=1) / denom
  valid = (denom.squeeze(-1) > self.eps).to(object_states.dtype).unsqueeze(-1)
  aggregated = aggregated_attn * valid + aggregated_mean * (1.0 - valid)
  ```

## 预期影响
- 提升主体对话更新的针对性，减少噪声带来的过平滑。
- 在负样本多、候选多的场景下提升稳定性与准确度，尤其对少数正样本更敏感。

## 可能的扩展
- 注意力温度参数以控制权重尖锐度。
- 使用双线性或可学习的匹配函数替换 MLP。
- 将候选增强输出或关系得分作为注意力先验；引入类型约束（NER 兼容性）。

## 风险与回退策略
- 若发现训练不稳定或过拟合，可降低注意力模型复杂度、加入正则或回退到均值聚合。
- 保持回退均值路径，避免掩码为空导致数值异常。

## 结论
用注意力替代均值聚合能让多轮对话更具选择性和稳健性，符合“主体与重要客体深入协商”的直觉，有助于提升联合抽取任务的表现。

