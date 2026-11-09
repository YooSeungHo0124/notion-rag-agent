# Tool Calling과 로컬 LLM 라우터 설계

- 작성일: 2026-11-10
- 목적: 공고에서 반복 요구하는 **"Tool / Function Calling 직접 구현"**(R5), **"Planning, Multi-step"**(R9), **"평가"**(R12)를 내 코드로 증명합니다.
- 배경: `05_스택_재검토.md`에서 `claude -p`에게 도구 실행을 맡기지 않기로 했습니다. 그래서 도구 호출 루프를 직접 만듭니다.

---

## 0. 결정 (재검토 반영): 로컬 LLM 라우터는 **1차 범위에서 제외**, 구조만 열어 둠
- **실용성:** 사용자 1명, 도구 약 8개 규모입니다. Claude가 한 번의 호출로 라우팅과 도구 선택을 함께 할 수 있어, 로컬 라우터를 추가해 얻는 이득(호출 수 절감)이 작습니다. 반면 VRAM이 약 14/16GB까지 차고, llama.cpp 서버를 하나 더 운영해야 하는 비용이 생깁니다.
- **어필 측면:** 소형 모델 라우터(Qwen3-4B LoRA 파인튜닝 + 분류 헤드 + GGUF + GBNF)는 이미 실무 프로젝트에서 운영 수준으로 구현한 경험이 있습니다. 토이 프로젝트에서 반복해도 새로 증명되는 역량이 거의 없습니다. → 이력서에는 실무 경험으로 쓰는 편이 더 강합니다.
- **1차에서 할 것:**
  - Tool 레지스트리, 실행기, 승인 게이트, Claude ReAct 루프 (아래 3~5절) → **tool calling 루프 직접 구현**
  - `Router` 인터페이스만 정의하고, 기본 구현은 Claude 라우팅 + 규칙 기반 빠른 경로로 둡니다.
  - 필요하면 이미 상주하는 임베딩 모델로 **의도 유사도 라우터**를 만듭니다 (추가 VRAM 없음).
- **2차 후보:** 사용량 한도나 지연이 실제 병목으로 측정되면 그때 로컬 라우터를 붙이고, 도입 전후를 비교합니다.
- 아래 1·2·6절의 로컬 라우터 내용은 2차 참고용으로 남겨 둡니다.

## 1. 구조: 2단계 (로컬 라우터 → Claude)

```
사용자 질의
   │
   ▼
[Tier 1] 로컬 라우터 LLM  (llama.cpp 서버, OpenAI 호환 API, 네이티브 tool calling)
   │  - 빠르고(수백 ms) 무료, 사용량 한도 없음
   │  - 출력: tool_call(name, args)  또는  route("claude_reasoning" | "direct")
   ▼
[Tool Executor]  ← 내가 구현 (스키마 검증, 실행, 에러 처리, 승인 게이트, 트레이싱)
   │  read 도구 : search_notes, get_timeline, get_page, list_db_items
   │  write 도구: create_page, update_page, add_db_item, archive_page  → interrupt(승인)
   ▼
[Tier 2] Claude (claude -p, 도구 없이)
   │  - 여러 검색 결과 종합, 회고, 긴 글 작성처럼 품질이 중요한 추론
   │  - 추가 도구가 필요하면 JSON action을 출력 → 내 ReAct 루프가 다시 Executor 호출
   ▼
답변 (+ 출처)
```

**두 단계로 나눈 이유**
- `claude -p`는 호출할 때마다 수 초가 걸리고 구독 사용량 한도도 있습니다. 반면 라우팅이나 단순 조회는 작은 모델로 충분합니다.
- 비용과 지연에 따라 모델을 나눠 쓰는 설계는 공고의 "프로덕션 운영" 요건에 대한 답이 됩니다.
- 로컬 모델의 **네이티브 tool calling**(OpenAI 형식의 `tools` / `tool_calls`)과 Claude의 **JSON action 방식**을 둘 다 구현하게 됩니다.

## 2. 로컬 라우터 모델 선정

| 후보 | 크기 (Q4 양자화 VRAM) | tool calling | 비고 |
|---|---|---|---|
| **Qwen3-4B-Instruct** | 약 3GB | 네이티브 지원 (Hermes 형식) | **기본 후보.** 성능과 크기의 균형이 좋음 |
| Qwen3-1.7B | 약 1.5GB | 지원 | VRAM이 부족할 때 대안 |
| Qwen3-8B | 약 5.5GB | 지원 | 정확도 상한. 임베딩 모델과 동시에 올리기는 빠듯함 |
| Llama-3.2-3B / Gemma 계열 | 약 2~3GB | 지원 | 한국어 품질은 Qwen이 유리할 것으로 예상 |

- 서빙: **llama.cpp `llama-server --jinja`** (OpenAI 호환 `/v1/chat/completions` + `tools`)
  - 대안 1: GBNF 문법으로 출력 JSON 형식을 강제합니다. 소형 모델의 형식 오류를 0으로 만들 수 있습니다.
  - 대안 2: Ollama
- VRAM 예산: Qwen3-Emb-4B(약 8GB) + bge-m3(약 2GB) + 리랭커(약 1GB) + 라우터 Qwen3-4B Q4(약 3GB) ≈ **14GB / 16GB**
  - 빠듯하면 라우터를 1.7B로 낮추거나, 리랭커를 필요할 때만 로딩합니다.
  - 질의 임베딩만 작은 모델로 바꾸는 방법은 쓸 수 없습니다. 질의와 인덱스는 같은 임베딩 모델이어야 합니다.
- "JEV" 모델: 정확한 정체를 확인하지 못해 후보에서 일단 제외했습니다. 확인되면 같은 평가셋으로 비교합니다.

## 3. Tool 레지스트리 (내 구현의 핵심)

```python
class SearchNotesArgs(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20)
    date_from: date | None = None
    tags: list[str] = []

@tool(name="search_notes", kind="read",
      desc="내 노션 기록을 의미+키워드 하이브리드로 검색")
def search_notes(args: SearchNotesArgs) -> list[Hit]: ...

@tool(name="create_page", kind="write", desc="노션에 새 페이지 생성")
def create_page(args: CreatePageArgs) -> PageRef: ...
```
- **하나의 정의에서 여러 형식을 자동 생성합니다:** pydantic 모델 → JSON Schema → ① OpenAI `tools` 형식(로컬 라우터) ② Claude 프롬프트용 도구 설명 ③ MCP 도구 (`my-memory` 서버도 같은 레지스트리를 재사용)
- **Executor가 담당하는 일:**
  - 인자 검증: 실패하면 에러 메시지를 모델에 돌려줘 재시도하게 합니다 (최대 2회).
  - 타임아웃 처리
  - `kind="write"`면 LangGraph `interrupt`로 미리보기를 보여주고 승인받습니다.
  - 멱등키 적용
  - 최대 스텝 제한 (무한 루프 방지)
  - 호출마다 OTel span 기록 (도구, 인자, 지연시간, 결과 크기)
- **삭제 도구는 만들지 않습니다.** 대신 `archive_page`(보관)만 제공해 되돌릴 수 있게 합니다.

## 4. LangGraph 흐름

```
START → rewrite_query → router(local)
          ├─ direct           → respond
          ├─ tool_call(read)  → execute → (결과 충분?) → respond / claude_reason
          ├─ tool_call(write) → propose → interrupt(승인) → execute → respond
          └─ claude_reasoning → claude_reason ⇄ execute (ReAct, 최대 N스텝) → respond
```

## 5. 평가 (어필 포인트)

| 지표 | 정의 |
|---|---|
| **Tool 선택 정확도** | 정답 도구를 골랐는가 |
| **인자 정확도** | 필수 인자가 정확히 일치하는가, 날짜·태그 추출이 맞는가 |
| 형식 유효성 | JSON 파싱과 스키마 검증 통과율 (GBNF 사용 전후 비교) |
| 라우팅 정확도 | direct / tool / claude 분기가 맞았는가 |
| 지연시간, Claude 호출 절감률 | 로컬 라우터 도입 전후 비교 |
| 안전성 | 쓰기 요청이 승인 없이 실행된 건수 (**목표 0**) |

- 라우터 평가셋: 질의 100~200개에 `{정답 경로, 도구, 인자}` 라벨을 붙입니다. Claude로 초안을 만들고 내가 검수합니다.
- 비교 실험: Qwen3-1.7B / 4B / 8B, GBNF 유무, few-shot 유무

## 6. 확장 (선택, 강력한 어필): 라우터 LoRA 파인튜닝
- 평가셋과 분리된 학습용 질의-도구 데이터를 만들어 Qwen3-1.7B나 4B를 LoRA로 미세조정합니다.
- 이후 GGUF로 변환해 llama.cpp에서 서빙합니다.
- 이것으로 "**작은 모델을 파인튜닝해 라우터 정확도를 X%에서 Y%로, 지연을 Z ms로**"라는 결과를 보여줄 수 있습니다.
- 기존에 갖춘 원격 GPU 서버의 LoRA 학습 워크플로를 그대로 활용할 수 있습니다.

## 7. 공고 요구사항 매핑 업데이트
| 요구사항 | 구현 |
|---|---|
| R5 Tool calling | 레지스트리 + Executor + 네이티브 tool calling(로컬) + JSON action(Claude) |
| R9 Planning / Multi-step | Claude ReAct 루프, 스텝 제한 |
| R11 안전한 실행 | 쓰기 도구 승인, archive만 허용, 멱등키 |
| R12 평가 | 도구 선택·인자 정확도, 라우팅 정확도 |
| R18 운영 (비용·지연) | 2단계 라우팅으로 Claude 호출 절감, 지연 측정 |
| (추가) 모델 경량화·파인튜닝 | 라우터 LoRA, GGUF 변환 |
