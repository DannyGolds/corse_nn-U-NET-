import os

def build_data_pairs(tp_dir, gt_dir):
    """
    Сканирует директории и находит пары (манипуляция <-> маска groundtruth)
    на основе вхождения имени подстроки.
    """
    # Проверяем существование папок
    if not os.path.exists(tp_dir) or not os.path.exists(gt_dir):
        raise ValueError(f"Путь неверный! Одна из папок не найдена:\nTP: {tp_dir}\nGT: {gt_dir}")

    tp_files = [f for f in os.listdir(tp_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    pairs = []
    
    print("Сканирование датасета и сопоставление пар...")
    for gt_name in gt_files:
        # Извлекаем ключ до '_gt'
        if '_gt' in gt_name:
            base_key = gt_name.split('_gt')[0].split("_")[-1] # Берем последний элемент после разделения по '_'
        else:
            continue
            
        # Ищем файл в папке Tp, в имени которого содержится этот ключ
        matched_tp = None
        for tp_name in tp_files:
            if base_key in tp_name:
                matched_tp = tp_name
                break
                
        if matched_tp:
            pairs.append({
                'tp_path': os.path.join(tp_dir, matched_tp),
                'gt_path': os.path.join(gt_dir, gt_name)
            })
            
    print(f"Количество успешно сопоставленных пар для сегментации: {len(pairs)}")
    return pairs