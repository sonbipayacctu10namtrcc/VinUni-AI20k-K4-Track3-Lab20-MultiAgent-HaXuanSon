# Design Template

## Problem

Xây dựng một research assistant nhận một câu hỏi nghiên cứu dạng tự do (vd. "Research
GraphRAG state-of-the-art"), tìm nguồn liên quan trên web, tổng hợp và đối chiếu các
nguồn đó, rồi viết một câu trả lời có trích dẫn `[n]` trỏ về đúng nguồn — đồng thời đo
được latency, cost, và citation coverage để so sánh với một baseline đơn giản hơn.

## Why multi-agent?

Single-agent (baseline) chỉ có kiến thức tham số của model, không có tool gọi search,
nên không thể trích dẫn nguồn thật — nó buộc phải trả lời từ trí nhớ và có thể bịa hoặc
lỗi thời. Multi-agent tách việc thành các bước có thể kiểm chứng độc lập: Researcher chỉ
chịu trách nhiệm lấy evidence thật (qua Tavily), Analyst chỉ so sánh/đánh giá evidence đó
(không tự bịa thêm claim), Writer chỉ tổng hợp có trích dẫn, Critic kiểm tra citation
coverage sau khi Writer xong. Việc tách vai trò cho phép: (1) retry đúng bước bị lỗi thay
vì retry toàn bộ, (2) đo latency/cost per-step, (3) chặn output không có citation trước
khi coi là "done". Đổi lại, pipeline chậm hơn ~2-3x và tốn nhiều lệnh gọi LLM hơn (xem
`reports/benchmark_report.md`) — đây là đánh đổi có chủ đích, không phải chi phí ẩn.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Đọc `ResearchState`, chọn route tiếp theo dựa trên field nào còn thiếu; enforce `max_iterations` | `ResearchState` | `route_history` entry mới + `iteration` | Không tự fail; nếu chạm `max_iterations` mà chưa xong, ghi lỗi vào `state.errors` và ép route `done` |
| Researcher | Gọi `SearchClient` (Tavily, hoặc mock offline nếu thiếu `TAVILY_API_KEY`), điền `sources` + `research_notes` đánh số `[n]` | `state.request.query` | `state.sources`, `state.research_notes` | Tavily lỗi/rỗng → `AgentExecutionError` → node wrapper ghi vào `state.errors`, field vẫn trống → Supervisor tự retry researcher tới khi hết `max_iterations` |
| Analyst | Đọc `research_notes`, rút key claims, so sánh nguồn, gắn cờ evidence yếu | `state.research_notes` | `state.analysis_notes` | LLM call lỗi (sau 3 lần retry của `LLMClient`) → như trên, Supervisor retry Analyst |
| Writer | Tổng hợp `research_notes` + `analysis_notes` thành câu trả lời có `[n]`, tự thêm mục "Sources" liệt kê toàn bộ `state.sources` | `research_notes`, `analysis_notes` | `state.final_answer` | Như Analyst; ngoài ra nếu model không trích `[n]` nào, Critic ở bước sau sẽ bắt được |
| Critic | Đếm `[n]` hợp lệ trong `final_answer` so với `len(sources)`, tính citation coverage, gắn cờ index bịa | `state.final_answer`, `state.sources` | `state.critic_notes`, có thể thêm `state.errors` | Không gọi LLM (regex-only) nên gần như không fail; low coverage/invalid index chỉ được ghi nhận, không chặn `done` |

## Shared state

`core/state.py` — mỗi field tồn tại vì một agent cụ thể cần đọc/ghi nó, và vì Supervisor
dùng chính "field nào còn `None`/rỗng" làm routing signal (thay vì một biến trạng thái
enum riêng, để tránh state trôi khỏi dữ liệu thật):

- `request`, `iteration`, `route_history` — input gốc + lịch sử routing để debug thứ tự
  chạy và tính benchmark (đã có sẵn trong skeleton).
- `sources`, `research_notes` — output của Researcher; Supervisor dùng để quyết định có
  cần chạy lại Researcher không.
- `analysis_notes` — output của Analyst; input của Writer.
- `final_answer` — output của Writer; input của Critic; cũng là output cuối cùng in ra CLI.
- `critic_notes` (mới thêm) — output của Critic; Supervisor dùng để biết pipeline đã qua
  bước kiểm tra citation hay chưa trước khi route `done`.
- `agent_results`, `trace`, `errors` — audit trail: mỗi agent append `AgentResult` (kèm
  token/cost usage) và trace event kèm `duration_seconds`; `errors` tích luỹ mọi lỗi từ
  mọi agent (không bị agent sau ghi đè) để benchmark tính `failure_rate` và report liệt
  kê "Failure modes".

## Routing policy

```
                 ┌─────────────┐
        ┌───────►│  supervisor  │◄───────────────┐
        │        └──────┬───────┘                │
        │   route = field còn thiếu đầu tiên      │
        │   (researcher > analyst > writer >      │
        │    critic > done); nếu iteration >=      │
        │    max_iterations → done (ghi lỗi nếu    │
        │    chưa có final_answer)                 │
        │               │                          │
   ┌────┴────┐   ┌──────┴──────┐   ┌─────────┐   ┌─────────┐
   │researcher│   │   analyst   │   │  writer │   │ critic  │
   └────┬────┘   └──────┬──────┘   └────┬────┘   └────┬────┘
        └───────────────┴───────────────┴─────────────┘
                    (mỗi worker route thẳng
                     về supervisor sau khi chạy)
```

Implement trong `graph/workflow.py` bằng LangGraph `StateGraph(ResearchState)`:
supervisor là node duy nhất có conditional edges (`add_conditional_edges` đọc
`state.route_history[-1]`); 4 worker node còn lại đều có edge một chiều về lại
supervisor (`add_edge(node, "supervisor")`). Route `"done"` map sang `END`.

## Guardrails

- **Max iterations**: `Settings.max_iterations` (mặc định 6, từ `.env`/`MAX_ITERATIONS`),
  enforce trong `SupervisorAgent._decide`. Với 4 worker + 1 vòng route "done", pipeline
  "happy path" tốn 5 lượt supervisor — mặc định 6 để chừa đúng 1 lượt retry.
- **Timeout**: `Settings.timeout_seconds` (mặc định 60s) truyền vào `LLMClient` (HTTP
  timeout của OpenAI client) và `SearchClient` (HTTP timeout của Tavily request).
- **Retry**: `LLMClient._call` retry tối đa 3 lần với backoff hàm mũ (qua `tenacity`) cho
  lỗi `OpenAIError` tạm thời (timeout, 5xx, rate limit). `SearchClient` không tự retry ở
  tầng HTTP — thay vào đó dựa vào Supervisor route lại "researcher" (retry ở tầng
  workflow, không phải tầng network).
- **Fallback**: một worker lỗi không crash graph — `MultiAgentWorkflow._guarded` bắt
  `AgentExecutionError`, ghi vào `state.errors`, trả state nguyên trạng để Supervisor
  route lại đúng worker đó. Nếu vẫn lỗi tới khi hết `max_iterations`, Supervisor ép
  `done` với `final_answer` có thể vẫn trống — CLI/benchmark coi đây là failure
  (`failure_rate=1.0` khi `not final_answer or state.errors`).
- **Validation**: `CriticAgent` kiểm tra citation bằng regex `\[(\d+)\]` sau khi Writer
  xong — chỉ số citation không khớp `sources` hoặc coverage = 0 (khi có sources) đều bị
  ghi vào `state.errors`, xuất hiện trong report/notes dù không chặn hoàn thành run.

## Benchmark plan

3 query trong `configs/lab_default.yaml` (đã chạy thật, xem `reports/benchmark_report.md`):

| Query | Metric đo | Expected outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | latency, cost, citation coverage | Multi-agent chậm hơn baseline nhưng đạt citation coverage cao (mục tiêu ≥80%) nhờ có Researcher + Critic |
| "Compare single-agent and multi-agent workflows for customer support" | quality (heuristic 0-10), failure rate | Cả hai path failure_rate 0% nếu API keys hợp lệ; multi-agent quality cao hơn nhờ điểm citation |
| "Summarize production guardrails for LLM agents" | latency, cost | Dùng để so sánh latency/cost trên câu hỏi ngắn hơn — kỳ vọng multi-agent vẫn ~2-3x latency của baseline do pipeline tuần tự |

Kết quả thực đo (3/3 query, cả hai path 0% failure): baseline avg latency 6.40s / quality
6.0; multi-agent avg latency 14.76s / quality 10.0 / citation coverage 100%. Phân tích
đầy đủ (bao gồm các failure mode chưa xảy ra trong lần chạy này) nằm trong
`reports/benchmark_report.md`.
