# 🧠 Memory Brain

Human brain-inspired memory system for LLMs, written in Rust.

## 개요

인간의 뇌 구조를 모방한 기억 시스템입니다. LLM과 통합하여 개인화된 컨텍스트 기반 응답을 제공합니다.

## 특징

### 🧠 뇌 기반 메모리 구조

| 메모리 타입 | 설명 | 예시 |
|------------|------|------|
| **Working Memory** | 단기 기억, 7개 제한 (Miller's Law) | 현재 대화 컨텍스트 |
| **Episodic Memory** | 일화 기억, "언제 무엇을" | "어제 Rust 버그 수정함" |
| **Semantic Memory** | 의미 기억, 사실과 개념 | "Rust는 소유권으로 메모리 안전성 보장" |
| **Procedural Memory** | 절차 기억, 패턴과 습관 | "에러 처리시 Result 사용" |

### 📉 망각 곡선 (Ebbinghaus)

- 사용하지 않는 기억은 점진적으로 약해짐
- 자주 접근하는 기억은 강화됨
- `R = e^(-t/S)` 공식 기반

### 🔍 시맨틱 검색

- 키워드 기반 텍스트 검색
- 임베딩 기반 유사도 검색
- 하이브리드 랭킹

### 🤖 LLM 통합

- MLX-LM (Apple Silicon 최적화)
- Ollama (로컬)
- OpenAI API (클라우드)

## 설치

```bash
# 클론
git clone <repository>
cd memory-brain

# 빌드
cargo build --release

# (선택) MLX 기능 포함
cargo build --release --features mlx
```

### MLX-LM 설정 (선택)

```bash
# venv 생성 및 설치
python3 -m venv ~/.venvs/mlx-lm
source ~/.venvs/mlx-lm/bin/activate
pip install mlx-lm

# 모델 다운로드 (자동)
# 첫 실행시 Llama 3.2 1B 모델 다운로드됨
```

## 사용법

### 기본 명령어

```bash
# 메모리 저장
memory-brain store "Rust uses ownership for memory safety"
memory-brain store "Pattern: use Result for errors" --type procedural --tags "rust,patterns"

# 메모리 검색
memory-brain recall "rust memory"
memory-brain search "systems programming"  # 시맨틱 검색

# 메모리 관리
memory-brain list                    # 최근 메모리
memory-brain show <id-prefix>        # 상세 보기
memory-brain stats                   # 통계
memory-brain export memories.json    # 내보내기
```

### LLM 명령어

```bash
# 메모리 기반 질문
memory-brain ask "What do I know about Rust?"

# 대화형 채팅
memory-brain chat

# 텍스트에서 사실 추출
memory-brain learn "Claude Code is an AI assistant by Anthropic"

# 주제별 요약
memory-brain summarize "programming languages"
```

### 옵션

```bash
-q, --quiet       # 시작 메시지 숨김
-n, --limit N     # 결과 수 제한
-t, --type TYPE   # 메모리 타입 (semantic|episodic|procedural)
--tags TAG1,TAG2  # 태그 추가
```

## 아키텍처

```
┌─────────────────────────────────────────┐
│            CLI (main.rs)                │
├─────────────────────────────────────────┤
│         Brain (lib.rs)                  │
│  ┌─────────┬─────────┬─────────┐       │
│  │ Working │Episodic │Semantic │       │
│  │ Memory  │ Memory  │ Memory  │       │
│  └────┬────┴────┬────┴────┬────┘       │
│       │         │         │            │
│  ┌────┴─────────┴─────────┴────┐       │
│  │     Storage (SQLite)         │       │
│  └──────────────────────────────┘       │
├─────────────────────────────────────────┤
│  Embedding │ Forgetting │ Consolidate   │
│  (GloVe)   │  Curve     │   Logic       │
├─────────────────────────────────────────┤
│         LLM Integration                 │
│  (MLX-LM / Ollama / OpenAI)            │
└─────────────────────────────────────────┘
```

## 파일 구조

```
src/
├── lib.rs          # Brain 메인 구조체, 기억 통합
├── main.rs         # CLI 인터페이스
├── types.rs        # MemoryItem, MemoryType 등
├── working.rs      # 작업 기억 (7개 제한)
├── episodic.rs     # 일화 기억
├── semantic.rs     # 의미 기억
├── procedural.rs   # 절차 기억 (패턴)
├── storage.rs      # SQLite 저장소
├── consolidate.rs  # 기억 통합 로직
├── forgetting.rs   # 망각 곡선 (Ebbinghaus)
├── embedding.rs    # 임베딩 (Hash/TF-IDF/MLX)
├── glove.rs        # GloVe 임베딩 로더
└── llm.rs          # LLM 통합 (MLX-LM/Ollama/OpenAI)
```

## 핵심 개념

### Miller's Law (7±2)
작업 기억의 용량 제한. 한 번에 7개 정도의 항목만 유지.

### Ebbinghaus 망각 곡선
`R = e^(-t/S)`
- R: 기억 유지율
- t: 시간
- S: 기억 강도

### 기억 통합 (Consolidation)
중요한 단기 기억을 장기 기억으로 이동:
- 감정적 기억 → 항상 저장
- 강한 기억 (strength > 0.6) → 저장
- 반복 접근 (count > 3) → 저장

## 의존성

- `rusqlite` - SQLite 데이터베이스
- `serde` / `serde_json` - 직렬화
- `chrono` - 시간 처리
- `uuid` - 고유 ID
- `dirs` - 디렉토리 경로
- `mlx-rs` (선택) - MLX 바인딩

## 향후 계획

- [ ] 서버 모드 (모델 상시 로딩)
- [ ] 웹 UI
- [ ] 더 큰 LLM 모델 지원
- [ ] 벡터 DB 통합 (Qdrant)
- [ ] 멀티 유저 지원

## 라이선스

MIT

## 참고 자료

- [Miller's Law](https://en.wikipedia.org/wiki/The_Magical_Number_Seven,_Plus_or_Minus_Two)
- [Ebbinghaus Forgetting Curve](https://en.wikipedia.org/wiki/Forgetting_curve)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Human Memory Systems](https://en.wikipedia.org/wiki/Memory)
