import argparse
from src.pipeline import PipelineRunner

def main():
    parser = argparse.ArgumentParser(description="Запуск ML пайплайну")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Шлях до файлу конфігурації (наприклад: configs/config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        runner = PipelineRunner(config_path=args.config)
        runner.run()
    except Exception as e:
        print(f"\nКритична помилка під час виконання: {e}")

if __name__ == "__main__":
    main()