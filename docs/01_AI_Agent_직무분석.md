# AI Agent 개발자 직무 분석 (채용공고 조사)

- 조사일: 2026-11-10
- 조사 경로: 잡코리아, 원티드, 직행(zighang), freehire, SK Careers 공고 검색
- 한계: 사람인과 자소설닷컴은 웹 검색 결과에 공고가 나오지 않아 **직접 확인하지 못했습니다**. 각 공고 내용은 검색 결과에서 발췌한 것이라 원문과 다를 수 있습니다. 지원 전에는 원문을 다시 확인하세요.

---

## 1. 시장 개요

- 잡코리아에서 "ai agent"로 검색하면 약 **709건**이 나옵니다.
- 경력 3년 이상 공고가 대부분입니다. 신입·주니어를 뽑는 공고는 휴트리온, 한국ESG데이터, 이우소프트, 코코아소프트 정도로 소수입니다.
- 직무 유형은 크게 네 가지로 나뉩니다.
  1. **에이전트 코어 엔지니어**: planning, tool-calling, memory, 멀티에이전트 설계 (네이버, 네이버클라우드, 티맵)
  2. **AI Agent 백엔드**: LLM 기능을 서비스 API와 연동 (골든플래닛, 이우소프트)
  3. **업무자동화(RPA + LLM)**: 사내 업무 자동화 에이전트 (하이퍼정보, 코코아소프트)
  4. **평가·운영(LLMOps)**: 품질·보안 평가, 트레이싱 (SK)

## 2. 공고별 요약

| 회사 | 경력 | 주요업무 | 핵심 기술·요건 |
|---|---|---|---|
| 네이버 AI 에이전트 엔지니어 | 경력 | LLM 기반 planning, tool-calling, multi-step reasoning 구조 설계·구현 | LangChain/LangGraph/AutoGen 실무, RAG·벡터DB, 평가 파이프라인 / 우대: GraphRAG, 프로덕션 운영 |
| 네이버클라우드 AI Agent 개발 | 경력 | AI·백엔드·외부 서비스 연동 서버/API 구축 | 백엔드, 에이전트 프레임워크 |
| 티맵모빌리티 SW Engineer - AI Agent | 경력 | LangChain 기반 에이전트 아키텍처, 외부 API 연동 Tool-Using Agent | LangChain, Function Calling |
| 티시스아이티 AI엔지니어 | 경력 | LangGraph 워크플로, **MCP 서버** 설계·구현, Graph RAG | 프로덕션 RAG·멀티 LLM 에이전트 운영 경험 |
| 골든플래닛 AI Agent 백엔드 | 경력 | 에이전트 서비스 개발 | LangChain, LangGraph, Google ADK, MCP 서버 구축, 클라우드 |
| SK (SK Careers) | 경력 | 에이전트 품질·보안 평가 체계, Logging/Tracing/Evaluation 환경 구축 | 평가, 관측(Observability) |
| 빅스데이터 AI Agent 개발 리더 | 5~8년+ | 멀티에이전트 오케스트레이션, 기술 리딩 | 리더십, 코드 품질 관리 |
| 멋쟁이사자처럼 AI Agent 개발자 | - | 멀티에이전트 오케스트레이션, Function Calling, MCP | 에이전트 프레임워크 |
| 하이퍼정보 AI-Agent팀 | 경력 | AI 개발 + RPA | Python, LLM/RAG/VectorDB, Linux, Docker/K8s / 우대: RPA, 프롬프트·워크플로 설계 |
| 팜소프트 AI Agent 개발 엔지니어 | 3년+ (6,600~9,900만원) | AI Agent 개발 | - |
| 휴트리온 AI Agent 개발자 | 신입·경력 | AI Agent 개발 | LangChain, RAG, OpenAI API |
| 한국ESG데이터 AI Agent 개발자 | 신입 | 벡터DB·RAG 기반 검색/질의, Tool-Using Agent | LangChain |
| 이우소프트 AI Agent Engineer | 신입 | 업무별 Agent 및 HTTP API 개발, Tool Calling, 평가 데이터 기반 품질 개선 | TypeScript, Node.js |
| 코코아소프트 AI Agent 개발직 | 신입·1년+ | AI 자동화 RPA 개발 | RPA |

## 3. 요구 역량 정리

### 3.1 필수 (대부분의 공고에 등장)
| 영역 | 내용 |
|---|---|
| 언어 | **Python** (FastAPI/Django). 일부 공고는 TypeScript/Node.js |
| LLM 활용 | OpenAI/Claude 등 API, 프롬프트 엔지니어링, **Function/Tool Calling** |
| 에이전트 프레임워크 | **LangGraph**(가장 많이 등장), LangChain, AutoGen, LlamaIndex, Google ADK |
| RAG | 문서 청킹, 임베딩, **벡터DB**, 검색 파이프라인 |
| 에이전트 구성요소 | Planning, Tool-calling, **Memory**, Multi-step reasoning |
| 백엔드·배포 | REST API, Linux, **Docker**, Kubernetes, 클라우드 |

### 3.2 우대 및 차별화 요소
- **MCP(Model Context Protocol) 서버 구축**: 최근 공고에서 필수나 우대로 자주 등장합니다.
- **멀티에이전트 오케스트레이션**: 에이전트 간 협업, supervisor 패턴
- **GraphRAG / 지식그래프·온톨로지**
- **평가(Evaluation) 파이프라인**: LLM-as-judge, 평가셋 구축, 품질 개선 루프
- **Observability**: Logging, Tracing (LangSmith, Langfuse 등)
- **프로덕션 운영 경험**: 비용, 지연시간, 안정성 관리
- 폐쇄망·온프레미스 환경 (sLLM, vLLM 배포, MLOps)
- RPA 및 업무자동화 워크플로 설계

## 4. 요구되는 "구현 경험" 체크리스트

공고 문구를 토이 프로젝트에서 증명할 수 있는 기능 단위로 바꾸면 다음과 같습니다.

- [ ] LLM이 외부 도구·API를 호출하는 **Tool-Using Agent**
- [ ] LangGraph 기반 **상태 그래프 워크플로** (분기, 반복, Human-in-the-loop)
- [ ] 벡터DB를 쓰는 **RAG 검색/질의**
- [ ] 단기(대화)·장기(사용자) **메모리**
- [ ] **멀티에이전트** 구조 (Supervisor와 Worker)
- [ ] **MCP 서버**를 직접 만들고 에이전트에 연결
- [ ] **평가 파이프라인** (테스트셋, 자동 채점, 지표)
- [ ] **Tracing/Logging** 연동
- [ ] FastAPI 서빙과 **Docker** 배포

## 5. 토이 프로젝트 방향 시사점

- 위 체크리스트를 한 프로젝트에 담으면 공고 대부분의 요건을 증명할 수 있습니다.
  - 필수: Tool calling, LangGraph, RAG, FastAPI, Docker
  - 차별화: MCP, 멀티에이전트, 평가/트레이싱
- 결과물 자체보다 **"에이전트가 잘 동작한다는 것을 어떻게 측정했는가"**(평가 지표)가 경력 공고와 신입 공고를 가르는 기준입니다.

## 출처
- 잡코리아: [휴트리온](https://www.jobkorea.co.kr/Recruit/GI_Read/48220104), [코코아소프트](https://www.jobkorea.co.kr/Recruit/GI_Read/49621597), [팜소프트](https://www.jobkorea.co.kr/Recruit/GI_Read/50005558), [블루비트](https://m.jobkorea.co.kr/Recruit/GI_Read/49340891), [검색결과](https://www.jobkorea.co.kr/Search/?stext=ai+agent)
- 원티드: [빅스데이터](https://www.wanted.co.kr/wd/316640), [네이버클라우드](https://www.wanted.co.kr/wd/286403), [멋쟁이사자처럼](https://www.wanted.co.kr/wd/320477), [티맵모빌리티](https://www.wanted.co.kr/wd/292769)
- 직행: [네이버 AI 에이전트 엔지니어](https://zighang.com/recruitment/0451a237-7301-445f-adc6-6032b301b3e9), [한국ESG데이터](https://zighang.com/recruitment/9ce8bc4f-6185-4273-8cf2-dda7b1945058)
- freehire: [하이퍼정보](https://freehire.me/jobs/ai-agenttim-aigaebal-rpa-caeyong-gyeongryeog-haipeojeongbo-k3v65h6k), [티시스아이티](https://freehire.me/jobs/aienjinieo-tisiseuaiti-ozo5eanh), [골든플래닛](https://freehire.me/jobs/ai-agent-baegendeugaebal-goldeunpeulraenis-wxvojzpb), [이우소프트](https://freehire.me/jobs/ai-agent-engineer-llm-typescript-sinib-iusopeuteu-xcu7btoj)
- [SK Careers](https://www.skcareers.com/Recruit/Detail/R261936)
