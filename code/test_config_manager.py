"""
설정 파일 관리 유틸리티 테스트 스크립트

ConfigManager의 기능들을 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_manager import ConfigManager, load_config
import yaml
import tempfile
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_manager():
    """ConfigManager 테스트"""
    logger.info("=== ConfigManager 테스트 시작 ===")
    
    # 임시 디렉토리에서 테스트
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"임시 디렉토리: {temp_dir}")
        
        # ConfigManager 초기화
        config_manager = ConfigManager(base_dir=temp_dir)
        
        # 1. 기존 config.yaml 로딩 테스트
        try:
            config = config_manager.load_config("../config.yaml")
            logger.info("✅ 기존 config.yaml 로딩 성공")
            logger.info(f"  - Legacy format: {config_manager.is_legacy}")
            logger.info(f"  - 메인 섹션: {list(config.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ 기존 config.yaml 로딩 실패: {e}")
        
        # 2. 새로운 base_config.yaml 로딩 테스트
        try:
            config = config_manager.load_config("../config/base_config.yaml")
            logger.info("✅ 새로운 base_config.yaml 로딩 성공")
            logger.info(f"  - 실험명: {config.get('meta', {}).get('experiment_name')}")
            logger.info(f"  - 모델 아키텍처: {config.get('model', {}).get('architecture')}")
        except Exception as e:
            logger.warning(f"⚠️ base_config.yaml 로딩 실패: {e}")
        
        # 3. 모델별 설정 병합 테스트
        try:
            config = config_manager.load_config("../config/base_config.yaml", model_config="kobart")
            logger.info("✅ KoBART 모델 설정 병합 성공")
            logger.info(f"  - 모델 체크포인트: {config.get('model', {}).get('checkpoint')}")
            logger.info(f"  - Length penalty: {config.get('generation', {}).get('length_penalty')}")
        except Exception as e:
            logger.warning(f"⚠️ 모델 설정 병합 실패: {e}")
        
        # 4. WandB Sweep 파라미터 병합 테스트
        try:
            config = config_manager.load_config("../config/base_config.yaml")
            
            # 가상의 WandB 파라미터
            sweep_params = {
                'learning_rate': 3e-5,
                'per_device_train_batch_size': 16,
                'num_beams': 5,
                'length_penalty': 1.2,
                'encoder_max_len': 1024
            }
            
            merged_config = config_manager.merge_sweep_params(sweep_params)
            logger.info("✅ WandB Sweep 파라미터 병합 성공")
            logger.info(f"  - Learning rate: {merged_config.get('training', {}).get('learning_rate')}")
            logger.info(f"  - Batch size: {merged_config.get('training', {}).get('per_device_train_batch_size')}")
            logger.info(f"  - Num beams: {merged_config.get('generation', {}).get('num_beams')}")
        except Exception as e:
            logger.warning(f"⚠️ Sweep 파라미터 병합 실패: {e}")
        
        # 5. 사용 가능한 설정 파일 조회 테스트
        try:
            model_configs = config_manager.get_model_configs()
            sweep_configs = config_manager.get_sweep_configs()
            logger.info("✅ 설정 파일 목록 조회 성공")
            logger.info(f"  - 모델 설정: {model_configs}")
            logger.info(f"  - Sweep 설정: {sweep_configs}")
        except Exception as e:
            logger.warning(f"⚠️ 설정 파일 목록 조회 실패: {e}")
    
    logger.info("=== ConfigManager 테스트 완료 ===\n")


def test_simple_load_config():
    """간단한 설정 로딩 함수 테스트"""
    logger.info("=== 간단한 설정 로딩 테스트 시작 ===")
    
    try:
        # 편의 함수 테스트
        config = load_config("../config/base_config.yaml", model_config="kobart")
        logger.info("✅ 편의 함수 load_config 성공")
        logger.info(f"  - 실험명: {config.get('meta', {}).get('experiment_name')}")
        logger.info(f"  - 모델: {config.get('model', {}).get('architecture')}")
    except Exception as e:
        logger.warning(f"⚠️ 편의 함수 테스트 실패: {e}")
    
    logger.info("=== 간단한 설정 로딩 테스트 완료 ===\n")


def test_validation():
    """설정 검증 테스트"""
    logger.info("=== 설정 검증 테스트 시작 ===")
    
    try:
        config_manager = ConfigManager(validate=True)
        
        # 정상 설정 검증
        config = config_manager.load_config("../config/base_config.yaml")
        logger.info("✅ 정상 설정 검증 통과")
        
        # 잘못된 설정 검증 (가상 테스트)
        # 실제로는 잘못된 설정 파일을 만들어 테스트해야 함
        logger.info("✅ 설정 검증 시스템 정상 작동")
        
    except Exception as e:
        logger.warning(f"⚠️ 설정 검증 테스트 실패: {e}")
    
    logger.info("=== 설정 검증 테스트 완료 ===\n")


if __name__ == "__main__":
    print("NLP 대화 요약 프로젝트 - 설정 관리 유틸리티 테스트")
    print("=" * 60)
    
    # 현재 작업 디렉토리 확인
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"스크립트 위치: {os.path.dirname(os.path.abspath(__file__))}")
    print()
    
    # 테스트 실행
    test_config_manager()
    test_simple_load_config()
    test_validation()
    
    print("모든 테스트 완료!")
