# setup_oai_testbed.sh
#!/bin/bash
set -e

echo "=== Setting up OAI + USRP Testbed for TCCA Evaluation ==="

# 1. Install OpenAirInterface
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
git checkout 2024.w46
./build_oai -I -w USRP --eNB --UE --install-system-files -c Debug

# 2. Configure 5G Core (Docker)
cd docker
docker-compose -f docker-compose-5g-core.yml up -d

# 3. Configure gNB (IAB Donor)
cd ../cmake_targets
./oai_gnb -C ../../openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
          --sa --rfsim 0 \
          --log_config.global_log_options=info

# 4. Configure IAB-Node (second USRP)
# Use OAI's IAB branch or patch standard gNB with backhaul config
./oai_gnb -C ../../openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/iab-node.band78.fr1.106PRB.usrpb210.conf \
          --sa --rfsim 0 \
          --parent-gnb-ip <DONOR_IP> \
          --log_config.global_log_options=info

# 5. Start TCCA rApp prototype
cd /path/to/tcca_pytorch
python3 tcca_rapp.py --config oai_testbed_config.yaml \
                     --telemetry-source oai_e2_adapter \
                     --inference-interval 100ms

echo "=== Testbed Ready ==="
echo "Donor gNB: <IP_DONOR>"
echo "IAB-Node: <IP_IAB>"
echo "TCCA rApp: <IP_TCCA>"
