import os
import urllib.parse
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import ChatOllama

# =========================================================
# 1. MariaDB 접속 정보 입력 (여기를 수정하세요)
# =========================================================
DB_USER =         # DB 아이디
DB_PASSWORD =   # DB 비밀번호 (특수문자 포함돼도 OK)
DB_HOST =      # DB IP 주소 (로컬이면 localhost)
DB_PORT =         # MariaDB 포트 (기본 3306)
DB_NAME =        # 접속할 데이터베이스 이름

# [중요] 비밀번호에 특수문자(@ 등)가 있으면 에러가 납니다. URL 인코딩 처리
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# MariaDB 연결 주소 생성 (charset=utf8mb4 옵션 추가: 한글 깨짐 방지)
mariadb_uri = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

print(f"📡 MariaDB({DB_HOST})에 연결 시도 중...")

# =========================================================
# 2. 엔진 연결 및 테스트
# =========================================================
try:
    engine = create_engine(mariadb_uri)
    
    # include_tables=['accidents']: 특정 테이블만 지정해서 가져오기 (권장)
    # sample_rows_in_table_info=3: LLM에게 데이터 샘플 3줄 보여주기 (정확도 상승)
    db = SQLDatabase(engine, sample_rows_in_table_info=3)
    
    print("✅ 연결 성공!")
    print(f"📂 인식된 테이블 목록: {db.get_usable_table_names()}")
    
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    print("팁: 아이디/비번, 방화벽(3306 포트), DB명이 맞는지 확인하세요.")
    exit()

# =========================================================
# 3. 로컬 LLM (Gemma 3) 설정
# =========================================================
llm = ChatOllama(
    model="gemma3:27b",  # 사용하시는 로컬 모델명
    temperature=0,
    base_url="http://localhost:11434"
)

# =========================================================
# 4. SQL Agent 생성
# =========================================================
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True, # 생각하는 과정 출력
    agent_type="zero-shot-react-description",
    handle_parsing_errors=True # 로컬 모델 에러 자동 수정
)

# =========================================================
# 5. 질문 실행
# =========================================================
# 실제 테이블에 있는 내용으로 질문을 바꿔보세요.
query = "이순신의 주소와 성별을 한국어로 알려줘"

print(f"\n💬 질문: {query}\n" + "="*50)

try:
    result = agent_executor.invoke(query)
    print("="*50)
    print(f"\n🚀 최종 답변: {result['output']}")
except Exception as e:
    print(f"⚠️ 에러 발생: {e}")
