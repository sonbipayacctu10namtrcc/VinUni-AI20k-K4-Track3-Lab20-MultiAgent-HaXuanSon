# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

Done: `cli.py`'s `baseline` command và `services/llm_client.py` gọi OpenAI thật (xem
`docs/design_template.md` và `reports/benchmark_report.md` cho số liệu latency/cost).

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

Done: xem "Routing policy" trong `docs/design_template.md` cho graph đầy đủ.

Gợi ý câu hỏi thiết kế — trả lời:

- Khi nào gọi Researcher? Khi `sources` hoặc `research_notes` còn trống.
- Khi nào gọi Analyst? Khi `research_notes` đã có nhưng `analysis_notes` còn trống.
- Khi nào gọi Writer? Khi `analysis_notes` đã có nhưng `final_answer` còn trống.
- Khi nào stop? Sau khi Critic chạy xong (`critic_notes` không còn `None`), hoặc khi
  `iteration >= max_iterations` (ép dừng, ghi lỗi nếu chưa có `final_answer`).
- Nếu agent fail thì retry hay fallback? Retry cùng agent đó ở lượt supervisor kế tiếp
  (vì field nó phụ trách vẫn trống) — không có agent thay thế; nếu vẫn fail tới khi hết
  `max_iterations` thì dừng với `state.errors` ghi rõ lý do thay vì lặp vô hạn.

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

Done: xem bảng "Agent roles" trong `docs/design_template.md`. Ngoài 3 worker trên, đã
thêm `agents/critic.py` (bonus, regex-only citation check) làm bước cuối trước `done`.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. **Case nào nên dùng multi-agent? Vì sao?** Khi câu trả lời bắt buộc phải có evidence
   kiểm chứng được (citation trỏ về nguồn thật) và task tự nhiên tách thành các bước có
   trách nhiệm khác nhau — ví dụ research report, so sánh nhiều nguồn có khả năng mâu
   thuẫn, hoặc bất kỳ pipeline nào cần một bước kiểm tra chất lượng độc lập trước khi trả
   kết quả (như Critic ở đây). Trong benchmark của chúng tôi, multi-agent đạt 100%
   citation coverage so với `n/a` (không thể) của baseline — đây chính là lý do tồn tại
   của kiến trúc này, không phải để "câu trả lời hay hơn" một cách chung chung.

2. **Case nào không nên dùng multi-agent? Vì sao?** Khi câu hỏi có thể trả lời tốt từ
   kiến thức tham số của model, không cần trích dẫn nguồn ngoài, và latency/cost quan
   trọng hơn khả năng kiểm chứng — ví dụ Q&A đơn giản, giải thích khái niệm phổ biến,
   hoặc bất kỳ tình huống nào cần phản hồi real-time. Benchmark thực đo cho thấy
   multi-agent chậm hơn baseline ~2.3x (14.76s vs 6.40s avg) và tốn nhiều lệnh gọi LLM
   hơn (Analyst + Writer + Researcher so với 1 lệnh gọi của baseline) — cho cùng một câu
   hỏi không cần trích dẫn, chi phí đó không đáng.
