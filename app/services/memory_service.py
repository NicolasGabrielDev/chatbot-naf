from typing import Dict, List

class MemoryService:
    def __init__(self) -> None:
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Adiciona uma mensagem ao histórico da sessão.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Retorna as últimas N rodadas de conversa para a sessão especificada.
        Uma rodada consiste tipicamente de uma mensagem do usuário e uma do assistente,
        então retornamos limit * 2 mensagens.
        """
        if session_id not in self.sessions:
            return []
        
        # limit * 2 para pegar tanto as perguntas do usuário quanto as respostas do bot
        return self.sessions[session_id][-(limit * 2):]

# Instância global do serviço para manter o estado em memória
memory_service = MemoryService()
