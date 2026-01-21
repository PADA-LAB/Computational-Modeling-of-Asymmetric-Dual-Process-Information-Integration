.PHONY: all run amazon coursera audible hotel \
        s1 s2 early-fusion residual late-fusion \
        clean


# 기본 전체 실행 (논문 전체 실험)
all:
	python -m src.main --config configs/runner.yaml
# 별칭
run: all

# 플랫폼별 전체 stage 실행
amazon:
	python -m src.main --platforms Amazon
coursera:
	python -m src.main --platforms Coursera
audible:
	python -m src.main --platforms Audible
hotel:
	python -m src.main --platforms Hotel

# Stage별 실행 (모든 플랫폼 기준)
s1:
	python -m src.main --stages s1
s2:
	python -m src.main --stages s2
early-fusion:
	python -m src.main --stages early-fusion
residual:
	python -m src.main --stages residual
late-fusion:
	python -m src.main --stages late-fusion

# 결과 삭제
clean:
	rm -rf outputs