"""State transition logic for TRACE environment."""

from datetime import datetime, timedelta
from trace.models import Observation, Action
from trace.scenarios import create_scenario


class Simulator:
    """Deterministic state transition engine."""
    
    def __init__(self, task_id: str, seed: int = 0):
        self.task_id = task_id
        self.seed = seed
        self.scenario = create_scenario(task_id, seed)
        self.current_obs = None
        self.current_time = None
        self.step_count = 0
        self.root_cause_revealed = {}  # tracks which causes have been revealed
    
    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self.scenario.reset()
        self.step_count = 0
        self.current_time = datetime.now()
        self.root_cause_revealed = {}
        
        # Generate initial state
        metrics = self.scenario.step()
        self.current_obs = self._build_observation(metrics)
        
        return self.current_obs
    
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one step.
        
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        self.current_time += timedelta(seconds=30)  # deterministic time advance
        
        # Check if action is valid/relevant
        is_relevant = self._is_action_relevant(action)
        solves_issue = self._does_action_solve(action)
        
        # Progress scenario
        # If remediation action, mark as "taken"
        remediation_actions = {
            "restart_service", "scale_workers", "restart_database",
            "rollback_release", "clear_queue"
        }
        resolved_action = action.action_type if (action.action_type in remediation_actions and solves_issue) else None

        metrics = self.scenario.step(resolved_action=resolved_action)
        
        # Handle inspection actions (reveal ground truth)
        inspection_msg = self._handle_inspection(action)
        
        incident_resolved = self.scenario.is_resolved()
        
        # Build observation
        self.current_obs = self._build_observation(metrics, inspection_msg)
        
        # Check terminal condition
        done = action.action_type in ["declare_healthy", "declare_unfixable"]
        
        info = {
            "step": self.step_count,
            "root_cause": self._get_root_cause(),
            "is_resolved": incident_resolved,
        }
        
        return self.current_obs, done, info
    
    def _build_observation(self, metrics: dict, inspection_msg: str = None) -> Observation:
        """Build observation from metrics and optional inspection."""
        # Determine service statuses based on metrics
        services = {
            "api_workers": self._get_service_status(
                metrics.get("cpu_usage_pct", 50),
                metrics.get("error_rate_pct", 0)
            ),
            "queue_service": self._get_service_status(
                metrics.get("queue_depth", 0) / 1000,
                metrics.get("error_rate_pct", 0)
            ),
            "database": self._get_service_status(
                metrics.get("memory_usage_pct", 50),
                metrics.get("api_latency_ms", 100) / 1000
            )
        }
        
        # Determine active alerts
        active_alerts = []
        if metrics.get("cpu_usage_pct", 0) > 75:
            active_alerts.append("alert_cpu_high")
        if metrics.get("error_rate_pct", 0) > 5:
            active_alerts.append("alert_high_error_rate")
        if metrics.get("queue_depth", 0) > 500:
            active_alerts.append("alert_queue_backlog")
        if metrics.get("api_latency_ms", 0) > 1000:
            active_alerts.append("alert_db_slow")
        if metrics.get("memory_usage_pct", 0) > 80:
            active_alerts.append("alert_pool_exhaustion")
        
        return Observation(
            timestamp=self.current_time.isoformat(),
            cpu_usage_pct=metrics.get("cpu_usage_pct", 50),
            memory_usage_pct=metrics.get("memory_usage_pct", 50),
            error_rate_pct=metrics.get("error_rate_pct", 0),
            api_latency_ms=metrics.get("api_latency_ms", 100),
            queue_depth=metrics.get("queue_depth", 0),
            services=services,
            active_alerts=active_alerts,
            last_inspection={"message": inspection_msg} if inspection_msg else None
        )
    
    def _get_service_status(self, cpu_factor: float, error_factor: float) -> str:
        """Map metrics to service status."""
        combined_health = (cpu_factor + error_factor) / 2
        
        if combined_health > 0.7:
            return "down"
        elif combined_health > 0.4:
            return "degraded"
        else:
            return "healthy"
    
    def _is_action_relevant(self, action: Action) -> bool:
        """Check if action is relevant to current incident."""
        # Inspection of relevant services/metrics is always somewhat relevant
        if action.action_type.startswith("inspect_"):
            return True
        
        # Remediation relevance depends on scenario
        if self.task_id == "easy_cpu_spike":
            return action.action_type == "scale_workers"
        elif self.task_id == "medium_cascade":
            return action.action_type in ["restart_service", "clear_queue"]
        elif self.task_id == "hard_mixed":
            return action.action_type in ["restart_database", "rollback_release"]
        
        return False
    
    def _does_action_solve(self, action: Action) -> bool:
        """Check if action actually solves the active problem."""
        root_cause = self._get_root_cause()
        
        if self.task_id == "easy_cpu_spike":
            return action.action_type == "scale_workers" and action.target == "api_workers"
        elif self.task_id == "medium_cascade":
            return action.action_type == "restart_service" and action.target == "queue_service"
        elif self.task_id == "hard_mixed":
            return action.action_type in ("restart_database", "rollback_release")
        
        return False
    
    def _handle_inspection(self, action: Action) -> str:
        """Handle inspection actions, reveal ground truth behind them."""
        if action.action_type == "inspect_logs":
            target = action.target or ""
            self.root_cause_revealed[f"logs_{target}"] = True
            
            if self.task_id == "easy_cpu_spike":
                if target == "api_workers":
                    return "Traffic spike detected. Request volume 5x normal. Consider scaling workers."
                return "No significant issues found in this service logs."
            elif self.task_id == "medium_cascade":
                if target == "queue_service":
                    return "Queue service memory usage is spiking. Heap allocation growing unbounded. Likely memory leak. May need restart."
                return "Service affected by upstream queue backlog."
            elif self.task_id == "hard_mixed":
                if target == "database":
                    return "Database queries have increased since recent release. New queries are inefficient and consuming connection pool."
                return "Service experiencing elevated errors due to database issues."
        
        elif action.action_type == "inspect_metrics":
            target = action.target or ""
            self.root_cause_revealed[f"metrics_{target}"] = True
            
            if target == "queue_depth":
                return "Queue depth critical and growing. Service memory may be leaking."
            elif target == "db_connections":
                return "Connection pool at 100% exhaustion. All connections in use. Restart needed."
            elif target == "cpu_usage_pct":
                if self.task_id == "easy_cpu_spike":
                    return "CPU usage elevated due to high request volume. Workers at capacity."
                return "CPU usage elevated but within expected range for current load."
            elif target == "memory_usage_pct":
                if self.task_id == "medium_cascade":
                    return "Memory usage growing unbounded in queue service. Leak suspected."
                return "Memory usage stable."
            elif target == "error_rate_pct":
                return "Error rate elevated. Correlates with service degradation."
            elif target == "api_latency_ms":
                if self.task_id == "hard_mixed":
                    return "Latency spike correlated with database connection issues."
                return "Latency elevated due to service degradation."
            return "Metric data returned. No anomalies detected for this metric."
        
        elif action.action_type == "inspect_alert":
            alert_id = action.target or ""
            self.root_cause_revealed[f"alert_{alert_id}"] = True
            
            if "pool" in alert_id:
                return "DB connection pool exhausted. All connections in use. Restart database to recover."
            elif "queue" in alert_id:
                return "Queue backlog growing due to service memory leak."
            elif "cpu" in alert_id:
                return "CPU high due to elevated traffic. Consider scaling workers."
            elif "error" in alert_id:
                return "Error rate elevated due to upstream service issues."
            elif "db_slow" in alert_id or "latency" in alert_id:
                return "Database responding slowly. Connection pool may be saturated."
            return "Alert acknowledged. No additional context available."
        
        return ""
    
    def _get_root_cause(self) -> str:
        """Get the root cause for this scenario."""
        if self.task_id == "easy_cpu_spike":
            return "traffic_spike"
        elif self.task_id == "medium_cascade":
            return "queue_memory_leak"
        elif self.task_id == "hard_mixed":
            return "db_connection_pool + release_regression"
        return "unknown"
    

