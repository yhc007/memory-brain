# CLS + Pekko Actor 통합 계획

## 개요

**목표**: Memory-Brain의 CLS(Complementary Learning System)를 Pekko Actor 시스템에 통합하여, LLM/Agent에게 인간과 유사한 기억 시스템 제공

**핵심 아이디어**: 
- 해마(Hippocampus) = 빠른 에피소드 기억 Actor
- 신피질(Neocortex) = 느린 의미 기억 통합 Actor
- Sleep/Dream = 백그라운드 통합 스케줄러

---

## 1. CLS 이론 배경

### 1.1 Complementary Learning System
```
┌─────────────────┐     빠른 학습     ┌─────────────────┐
│   Hippocampus   │ ◄──────────────► │   Experience    │
│  (에피소드 기억)  │                  │   (새로운 경험)   │
└────────┬────────┘                  └─────────────────┘
         │
         │ 느린 통합 (Sleep/Replay)
         ▼
┌─────────────────┐
│    Neocortex    │
│   (의미 기억)    │
└─────────────────┘
```

### 1.2 핵심 원리
1. **빠른 학습 (Hippocampus)**: 새 경험을 빠르게 저장, 높은 가소성
2. **느린 통합 (Neocortex)**: 패턴 추출, 일반화, 낮은 가소성
3. **Replay/Dream**: 해마 → 신피질로 기억 전이, 간섭 최소화

---

## 2. 현재 Memory-Brain 구조

### 2.1 기존 모듈
| 모듈 | 역할 | CLS 매핑 |
|------|------|----------|
| `hippocampus.rs` | 빠른 에피소드 저장 | Hippocampus |
| `episodic.rs` | 에피소드 기억 관리 | Hippocampus |
| `semantic_store.rs` | 의미 기억 저장 | Neocortex |
| `consolidate.rs` | 기억 통합 | Sleep Transfer |
| `dream.rs` | 꿈/리플레이 처리 | Replay |
| `forgetting.rs` | 망각 곡선 적용 | Memory Decay |

### 2.2 현재 저장소
- **CoreVecDB**: 벡터 저장소 (localhost:3100)
- **Embedding Server**: BGE-M3 (localhost:3201)

---

## 3. Pekko Actor 통합 설계

### 3.1 Actor 구조
```
                    ┌─────────────────────┐
                    │   MemoryGuardian    │
                    │   (Supervisor)      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ HippocampusActor│  │  NeocortexActor │  │   DreamActor    │
│                 │  │                 │  │                 │
│ - store()       │  │ - query()       │  │ - consolidate() │
│ - recall()      │  │ - associate()   │  │ - replay()      │
│ - forget()      │  │ - generalize()  │  │ - prune()       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     ▲                     │
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                         (Message Passing)
```

### 3.2 메시지 타입
```rust
// 기억 저장 요청
enum MemoryMessage {
    // Hippocampus
    Store { content: String, context: Context, reply_to: ActorRef },
    Recall { query: String, k: usize, reply_to: ActorRef },
    
    // Neocortex  
    Associate { memories: Vec<MemoryId>, reply_to: ActorRef },
    Generalize { pattern: Pattern, reply_to: ActorRef },
    
    // Dream
    StartConsolidation,
    Replay { memory_ids: Vec<MemoryId> },
    ApplyForgetting { threshold: f32 },
}
```

### 3.3 Actor 구현

#### HippocampusActor
```rust
pub struct HippocampusActor {
    store: CoreVecDBClient,
    embedding: EmbeddingClient,
    working_memory: VecDeque<Memory>, // 최근 N개
}

impl Actor for HippocampusActor {
    type Message = MemoryMessage;
    
    async fn receive(&mut self, msg: Self::Message) -> Result<()> {
        match msg {
            Store { content, context, reply_to } => {
                // 1. 임베딩 생성
                // 2. CoreVecDB에 저장
                // 3. working_memory에 추가
                // 4. reply_to에 응답
            }
            Recall { query, k, reply_to } => {
                // 1. 쿼리 임베딩
                // 2. 벡터 검색
                // 3. 결과 반환
            }
            _ => {}
        }
    }
}
```

#### DreamActor (백그라운드)
```rust
pub struct DreamActor {
    hippocampus: ActorRef<MemoryMessage>,
    neocortex: ActorRef<MemoryMessage>,
    schedule: CronSchedule, // 새벽 4시
}

impl Actor for DreamActor {
    async fn receive(&mut self, msg: Self::Message) -> Result<()> {
        match msg {
            StartConsolidation => {
                // 1. 최근 에피소드 가져오기
                // 2. 패턴 추출 (LLM 활용 가능)
                // 3. 의미 기억으로 통합
                // 4. 약한 연결 페이드아웃
                // 5. 강한 연결 강화
            }
            Replay { memory_ids } => {
                // 기억 리플레이로 연결 강화
            }
            ApplyForgetting { threshold } => {
                // Ebbinghaus 망각 곡선 적용
            }
        }
    }
}
```

---

## 4. 구현 단계

### Phase 1: Actor 기본 구조 (1주)
- [ ] `memory-actor` 크레이트 생성
- [ ] `HippocampusActor` 구현
- [ ] `NeocortexActor` 구현  
- [ ] `DreamActor` 구현
- [ ] `MemoryGuardian` (Supervisor) 구현

### Phase 2: 메시지 프로토콜 (3일)
- [ ] 메시지 타입 정의
- [ ] Actor 간 통신 구현
- [ ] 에러 핸들링 / 재시도 로직

### Phase 3: 통합 및 테스트 (1주)
- [ ] CoreVecDB 연동
- [ ] Embedding Server 연동
- [ ] 기존 CLI와 호환
- [ ] 통합 테스트

### Phase 4: 스케줄링 (3일)
- [ ] Dream 스케줄러 (새벽 4시)
- [ ] 실시간 망각 곡선 적용
- [ ] 메트릭 / 모니터링

---

## 5. 기대 효과

1. **확장성**: Actor 모델로 분산 처리 가능
2. **격리**: 각 메모리 시스템 독립 운영
3. **생물학적 유사성**: CLS 이론 기반 자연스러운 기억
4. **LLM 통합**: Agent가 장기 기억 활용 가능

---

## 6. 의존성

```toml
[dependencies]
pekko-actor = { path = "../pekko-rust/pekko-actor" }
memory-brain = { path = "." }
tokio = { version = "1", features = ["full"] }
```

---

## 7. 다음 단계

1. **이 계획서 검토** ← 현재
2. Phase 1 시작: Actor 기본 구조
3. 기존 memory-brain 코드 리팩토링
4. pekko-actor와 통합

---

*작성일: 2026-03-25*
*작성자: Sam 🦊 + Paul*
