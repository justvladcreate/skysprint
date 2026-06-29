#model interface
class PilotModel:
    def predict(self, state: dict) -> int:
        """Принимает словарь состояния, возвращает действие 0 (отпустить) или 1 (тянуть)."""
        raise NotImplementedError

    def save(self, filepath: str):
        pass

    @classmethod
    def load(cls, filepath: str) -> 'PilotModel':
        pass