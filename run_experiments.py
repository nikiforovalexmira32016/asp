import subprocess
import time
import os
import sys
from datetime import datetime
import glob

# Определяем путь к папке, где находится этот скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Переходим в эту папку

print(f"Рабочая папка: {SCRIPT_DIR}")

# Создаем папку для логов
log_dir = os.path.join(SCRIPT_DIR, "experiment_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Лог-файл с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"experiments_{timestamp}.log")

def log_message(msg):
    """Запись в лог-файл и вывод на экран"""
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Список экспериментов
experiments = [
    # ========== ЭКСПЕРИМЕНТ 1: Влияние β ==========
    (0.2, 0.2, 0.2, 4, "β=0.2 - затухающая эпидемия"),
    (0.4, 0.2, 0.2, 4, "β=0.4 - умеренная вспышка"),
    (0.6, 0.2, 0.2, 4, "β=0.6 - пороговый режим"),
    (0.8, 0.2, 0.2, 4, "β=0.8 - устойчивая эндемия"),
    (1.0, 0.2, 0.2, 4, "β=1.0 - высокая эндемия"),
    
    # ========== ЭКСПЕРИМЕНТ 2А: Влияние λ при низком β ==========
    (0.3, 0.2, 0.1, 4, "β=0.3, λ=0.1 - низкая активность"),
    (0.3, 0.2, 0.3, 4, "β=0.3, λ=0.3 - средняя активность"),
    (0.3, 0.2, 0.5, 4, "β=0.3, λ=0.5 - высокая активность"),
    (0.3, 0.2, 0.7, 4, "β=0.3, λ=0.7 - очень высокая активность"),
    
    # ========== ЭКСПЕРИМЕНТ 2Б: Влияние λ при высоком β ==========
    (0.9, 0.2, 0.1, 4, "β=0.9, λ=0.1 - низкая активность"),
    (0.9, 0.2, 0.3, 4, "β=0.9, λ=0.3 - средняя активность"),
    (0.9, 0.2, 0.5, 4, "β=0.9, λ=0.5 - высокая активность"),
    (0.9, 0.2, 0.7, 4, "β=0.9, λ=0.7 - очень высокая активность"),
    
    # ========== ЭКСПЕРИМЕНТ 3: Влияние δ ==========
    (0.8, 0.05, 0.3, 4, "δ=0.05 - очень медленное выздоровление"),
    (0.8, 0.1, 0.3, 4, "δ=0.1 - медленное выздоровление"),
    (0.8, 0.2, 0.3, 4, "δ=0.2 - среднее выздоровление"),
    (0.8, 0.4, 0.3, 4, "δ=0.4 - быстрое выздоровление"),
    (0.8, 0.7, 0.3, 4, "δ=0.7 - очень быстрое выздоровление"),
    
    # ========== ЭКСПЕРИМЕНТ 4: Влияние C ==========
    (0.6, 0.2, 0.3, 2, "C=2 - разреженная сеть"),
    (0.6, 0.2, 0.3, 4, "C=4 - средняя связность"),
    (0.6, 0.2, 0.3, 6, "C=6 - плотная сеть"),
    (0.6, 0.2, 0.3, 10, "C=10 - очень плотная сеть"),
]

def run_experiment(beta, delta, lambda_, C, description, N_max=100, T=100, n_runs=50):
    """
    Запуск одного эксперимента
    """
    log_message(f"\n{'='*60}")
    log_message(f"Эксперимент: {description}")
    log_message(f"Параметры: N_max={N_max}, C={C}, λ={lambda_}, β={beta}, δ={delta}, T={T}, прогонов={n_runs}")
    log_message(f"{'='*60}")
    
    # Полный путь к asp.py
    asp_path = os.path.join(SCRIPT_DIR, "asp.py")
    
    if not os.path.exists(asp_path):
        log_message(f"❌ ОШИБКА: Файл {asp_path} не найден!")
        log_message(f"   Текущая папка: {SCRIPT_DIR}")
        log_message(f"   Содержимое папки: {os.listdir(SCRIPT_DIR)}")
        return False
    
    try:
        # Формируем команду с параметрами как аргументы командной строки
        cmd = [
            sys.executable, asp_path,
            str(N_max), str(C), str(lambda_), str(beta), str(delta), str(T), str(n_runs)
        ]
        
        log_message(f"Команда: {' '.join(cmd)}")
        
        # Запускаем процесс
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=SCRIPT_DIR,
            encoding='utf-8',
            errors='ignore'
        )
        
        # Получаем результат
        stdout, stderr = proc.communicate(timeout=300)  # 5 минут таймаут
        
        # Сохраняем вывод в лог
        log_message("--- РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ---")
        if stdout:
            # Фильтруем вывод, убираем возможные проблемы с кодировкой
            clean_stdout = stdout.encode('utf-8', errors='ignore').decode('utf-8')
            log_message(clean_stdout)
        
        if stderr:
            clean_stderr = stderr.encode('utf-8', errors='ignore').decode('utf-8')
            log_message(f"!!! ОШИБКИ: {clean_stderr}")
            
        # Ищем созданные файлы
        files = [f for f in os.listdir(SCRIPT_DIR) 
                if f.startswith('aggregated_') and f.endswith('.csv')]
        if files:
            # Находим самый новый файл
            latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(SCRIPT_DIR, f)))
            log_message(f"✅ Создан файл: {latest_file}")
            
            # Переименовываем для наглядности
            exp_name = f"exp_beta{beta}_lambda{lambda_}_delta{delta}_C{C}"
            new_filename = f"{exp_name}_{latest_file}"
            
            # Проверяем, не существует ли уже такой файл
            new_path = os.path.join(SCRIPT_DIR, new_filename)
            old_path = os.path.join(SCRIPT_DIR, latest_file)
            
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                log_message(f"✅ Переименован в: {new_filename}")
            else:
                log_message(f"⚠️ Файл {new_filename} уже существует")
        
        # Переименовываем графики, если есть
        for f in glob.glob(os.path.join(SCRIPT_DIR, "unified_view_*.png")):
            exp_name = f"exp_beta{beta}_lambda{lambda_}_delta{delta}_C{C}"
            base_name = os.path.basename(f)
            new_name = os.path.join(SCRIPT_DIR, f"{exp_name}_{base_name}")
            
            if not os.path.exists(new_name):
                os.rename(f, new_name)
                log_message(f"✅ График сохранен: {os.path.basename(new_name)}")
            
        return True
        
    except subprocess.TimeoutExpired:
        proc.kill()
        log_message("❌ Таймаут: эксперимент занял слишком много времени")
        return False
    except Exception as e:
        log_message(f"❌ Ошибка: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False

def main():
    log_message(f"ЗАПУСК СЕРИИ ЭКСПЕРИМЕНТОВ")
    log_message(f"Рабочая папка: {SCRIPT_DIR}")
    log_message(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Всего экспериментов: {len(experiments)}")
    log_message(f"Лог-файл: {log_file}")
    
    # Проверяем наличие asp.py
    asp_path = os.path.join(SCRIPT_DIR, "asp.py")
    if not os.path.exists(asp_path):
        log_message(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл asp.py не найден в {SCRIPT_DIR}")
        log_message(f"   Убедитесь, что оба файла в одной папке:")
        log_message(f"   - {asp_path}")
        log_message(f"   - {os.path.join(SCRIPT_DIR, 'run_experiments.py')}")
        return
    
    successful = 0
    failed = 0
    
    for i, (beta, delta, lambda_, C, description) in enumerate(experiments, 1):
        log_message(f"\n📌 Эксперимент {i}/{len(experiments)}")
        
        success = run_experiment(
            beta=beta,
            delta=delta,
            lambda_=lambda_,
            C=C,
            description=description,
            N_max=100,
            T=100,
            n_runs=50  # Для теста можно 20, для статьи 200
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Пауза между экспериментами
        if i < len(experiments):
            log_message("⏳ Пауза 5 секунд...")
            time.sleep(5)
    
    # Итоги
    log_message(f"\n{'='*60}")
    log_message(f"СЕРИЯ ЭКСПЕРИМЕНТОВ ЗАВЕРШЕНА")
    log_message(f"Успешно: {successful}")
    log_message(f"С ошибками: {failed}")
    log_message(f"Результаты в папке: {SCRIPT_DIR}")
    log_message(f"Лог сохранен в: {log_file}")
    log_message(f"Время окончания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()