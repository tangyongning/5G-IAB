# inject_faults_oai.py
import subprocess
import time
import requests

class OAIFaultInjector:
    def __init__(self, donor_ip, iab_ip):
        self.donor_ip = donor_ip
        self.iab_ip = iab
        
    def inject_beam_misalignment(self, node_ip, phase_offset_deg=30):
        """Inject beam misalignment via OAI MAC config update"""
        # OAI supports runtime config via REST API (if enabled)
        payload = {
            "node": node_ip,
            "beamforming": {
                "phase_offset": phase_offset_deg,
                "apply_immediately": True
            }
        }
        resp = requests.post(f"http://{node_ip}:8080/api/v1/mac/config", json=payload)
        return resp.status_code == 200
    
    def inject_harq_failure(self, node_ip, nack_probability=0.5):
        """Force HARQ NACKs via OAI MAC scheduler patch"""
        # Requires custom OAI build with debug hooks
        cmd = f"ssh {node_ip} 'echo {nack_probability} > /proc/oai/mac/harq_nack_inject'"
        subprocess.run(cmd, shell=True)
        
    def inject_rf_interference(self, usrp_serial, power_dbm=20, freq_mhz=3500):
        """Use third USRP as jammer via GNU Radio"""
        cmd = f"python3 jammer.py --serial {usrp_serial} --freq {freq_mhz} --gain {power_dbm}"
        subprocess.Popen(cmd, shell=True)
        return True
    
    def inject_core_congestion(self, donor_ip, rate_mbps=10):
        """Rate-limit N3 interface to simulate donor congestion"""
        cmd = f"ssh {donor_ip} 'tc qdisc add dev n3if root tbf rate {rate_mbps}mbit burst 32k latency 400ms'"
        subprocess.run(cmd, shell=True)
        
    def clear_faults(self, node_ip):
        """Restore normal operation"""
        # Reset beamforming
        requests.post(f"http://{node_ip}:8080/api/v1/mac/config", 
                     json={"beamforming": {"reset": True}})
        # Clear tc rules
        subprocess.run(f"ssh {node_ip} 'tc qdisc del dev n3if root'", shell=True)
