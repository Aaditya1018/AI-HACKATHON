#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Running Campus Energy Forecast & Advisory Pipeline"
echo "=========================================================="
python3 run_pipeline.py
echo ""
echo "Execution completed successfully. Review reports/ and reports/figures/."
