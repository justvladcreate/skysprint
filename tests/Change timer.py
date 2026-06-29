from agent import NeuroevoAgent

path = "../models/level_8/Medium.pkl"
# Загружаем модель (на CPU)
agent = NeuroevoAgent.load(path, device='cpu')

# Меняем нужные поля
agent.pb_time = 39.40   # новое время в секундах
agent.deaths = 0      # новое количество смертей

# Сохраняем обратно (перезаписываем файл)
agent.save(path, pb_time=agent.pb_time, deaths=agent.deaths)