"""
simulationIPC通信模块
forFlask后端 and simulation脚本之间's process间通信

throughfile系统实现简单's command/responsemode:
1. Flask写入commandto  commands/ directory
2. simulation脚本round询commanddirectory, executecommand并写入responseto  responses/ directory
3. Flaskround询responsedirectoryGetresult
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger('mirofish.simulation_ipc')


class CommandType(str, Enum):
    """commandtype"""
    INTERVIEW = "interview"           # 单 itemsAgentinterview
    BATCH_INTERVIEW = "batch_interview"  # batchinterview
    CLOSE_ENV = "close_env"           # Closeenvironment


class CommandStatus(str, Enum):
    """commandstatus"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IPCCommand:
    """IPCcommand"""
    command_id: str
    command_type: CommandType
    args: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type.value,
            "args": self.args,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCCommand':
        return cls(
            command_id=data["command_id"],
            command_type=CommandType(data["command_type"]),
            args=data.get("args", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class IPCResponse:
    """IPCresponse"""
    command_id: str
    status: CommandStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCResponse':
        return cls(
            command_id=data["command_id"],
            status=CommandStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


class SimulationIPCClient:
    """
    simulationIPCclient(Flask端use)
    
    for向simulationprocesssendcommand并waitingresponse
    """
    
    def __init__(self, simulation_dir: str):
        """
        initializingIPCclient
        
        Args:
            simulation_dir: simulationdatadirectory
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # Ensuredirectoryexists
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def send_command(
        self,
        command_type: CommandType,
        args: Dict[str, Any],
        timeout: float = 60.0,
        poll_interval: float = 0.5
    ) -> IPCResponse:
        """
        sendcommand并waitingresponse
        
        Args:
            command_type: commandtype
            args: commandparameter
            timeout: timeouttime(seconds)
            poll_interval: round询间隔(seconds)
            
        Returns:
            IPCResponse
            
        Raises:
            TimeoutError: waitingresponsetimeout
        """
        command_id = str(uuid.uuid4())
        command = IPCCommand(
            command_id=command_id,
            command_type=command_type,
            args=args
        )
        
        # Writecommandfile
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"sendIPCcommand: {command_type.value}, command_id={command_id}")
        
        # Wait forresponse
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    response = IPCResponse.from_dict(response_data)
                    
                    # Clean upcommand and responsefile
                    try:
                        os.remove(command_file)
                        os.remove(response_file)
                    except OSError:
                        pass
                    
                    logger.info(f"收to IPCresponse: command_id={command_id}, status={response.status.value}")
                    return response
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"parseresponsefailed: {e}")
            
            time.sleep(poll_interval)
        
        # Timeout
        logger.error(f"waitingIPCresponsetimeout: command_id={command_id}")
        
        # Clean upcommandfile
        try:
            os.remove(command_file)
        except OSError:
            pass
        
        raise TimeoutError(f"waitingcommandresponsetimeout ({timeout}seconds)")
    
    def send_interview(
        self,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> IPCResponse:
        """
        send单 itemsAgentinterviewcommand
        
        Args:
            agent_id: Agent ID
            prompt: interviewquestion
            platform: 指定platform(optional)
                - "twitter": onlyinterviewTwitterplatform
                - "reddit": onlyinterviewRedditplatform  
                - None: dual-platformsimulationtimesimultaneouslyinterviewtwo itemsplatform, 单platformsimulationtimeinterviewthisplatform
            timeout: timeouttime
            
        Returns:
            IPCResponse, resultfieldcontainsinterviewresult
        """
        args = {
            "agent_id": agent_id,
            "prompt": prompt
        }
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_batch_interview(
        self,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> IPCResponse:
        """
        sendbatchinterviewcommand
        
        Args:
            interviews: interviewlist, each元素contains {"agent_id": int, "prompt": str, "platform": str(optional)}
            platform: defaultplatform(optional, 会被eachinterview items's platform覆盖)
                - "twitter": defaultonlyinterviewTwitterplatform
                - "reddit": defaultonlyinterviewRedditplatform
                - None: dual-platformsimulationtimeeachAgentsimultaneouslyinterviewtwo itemsplatform
            timeout: timeouttime
            
        Returns:
            IPCResponse, resultfieldcontainshasinterviewresult
        """
        args = {"interviews": interviews}
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.BATCH_INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_close_env(self, timeout: float = 30.0) -> IPCResponse:
        """
        sendcloseenvironmentcommand
        
        Args:
            timeout: timeouttime
            
        Returns:
            IPCResponse
        """
        return self.send_command(
            command_type=CommandType.CLOSE_ENV,
            args={},
            timeout=timeout
        )
    
    def check_env_alive(self) -> bool:
        """
        checksimulationenvironmentwhether to存活
        
        throughcheck env_status.json file来判断
        """
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        if not os.path.exists(status_file):
            return False
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return status.get("status") == "alive"
        except (json.JSONDecodeError, OSError):
            return False


class SimulationIPCServer:
    """
    simulationIPCserver(simulation脚本端use)
    
    round询commanddirectory, executecommand并returnresponse
    """
    
    def __init__(self, simulation_dir: str):
        """
        initializingIPCserver
        
        Args:
            simulation_dir: simulationdatadirectory
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # Ensuredirectoryexists
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # environmentstatus
        self._running = False
    
    def start(self):
        """标记serverrunningstatus"""
        self._running = True
        self._update_env_status("alive")
    
    def stop(self):
        """标记serverstopstatus"""
        self._running = False
        self._update_env_status("stopped")
    
    def _update_env_status(self, status: str):
        """Updateenvironmentstatusfile"""
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_commands(self) -> Optional[IPCCommand]:
        """
        round询commanddirectory, returnround one items待process's command
        
        Returns:
            IPCCommand  or  None
        """
        if not os.path.exists(self.commands_dir):
            return None
        
        # by time排序Getcommandfile
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return IPCCommand.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"readcommandfilefailed: {filepath}, {e}")
                continue
        
        return None
    
    def send_response(self, response: IPCResponse):
        """
        sendresponse
        
        Args:
            response: IPCresponse
        """
        response_file = os.path.join(self.responses_dir, f"{response.command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Deletecommandfile
        command_file = os.path.join(self.commands_dir, f"{response.command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def send_success(self, command_id: str, result: Dict[str, Any]):
        """Sendsuccessresponse"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.COMPLETED,
            result=result
        ))
    
    def send_error(self, command_id: str, error: str):
        """Senderrorresponse"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.FAILED,
            error=error
        ))
