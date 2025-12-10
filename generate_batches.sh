#!/bin/bash
# Batch Generation Script for Arabic EOU Dataset
# This script generates 1,200 conversations in 5 batches of 240 each

set -e  # Exit on error

# Configuration
TOTAL_CONVERSATIONS=1200
BATCH_SIZE=240
NUM_BATCHES=$((TOTAL_CONVERSATIONS / BATCH_SIZE))
DATA_DIR="data"
LOGS_DIR="logs"
COMBINED_OUTPUT="$DATA_DIR/arabic_eou_dataset_${TOTAL_CONVERSATIONS}.json"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$DATA_DIR" "$LOGS_DIR"

# Check if API key is set
if [ -z "$NEBIUS_API_KEY" ]; then
    echo -e "${YELLOW}Warning: NEBIUS_API_KEY environment variable not set${NC}"
    echo "Please set it with: export NEBIUS_API_KEY='your-api-key'"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Arabic EOU Dataset Generation${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total conversations: ${TOTAL_CONVERSATIONS}"
echo -e "Batch size: ${BATCH_SIZE}"
echo -e "Number of batches: ${NUM_BATCHES}"
echo -e "Output directory: ${DATA_DIR}"
echo -e "${GREEN}========================================${NC}\n"

# Generate batches
START_TIME=$(date +%s)

for i in $(seq 1 $NUM_BATCHES); do
    echo -e "${BLUE}Generating batch $i of $NUM_BATCHES...${NC}"
    
    python arabic_eou_data_generator.py \
        --num-conversations $BATCH_SIZE \
        --output-file "$DATA_DIR/batch_$i.json" \
        --log-file "$LOGS_DIR/batch_$i.log" \
        --validate
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Batch $i completed successfully${NC}\n"
    else
        echo -e "${YELLOW}✗ Batch $i failed${NC}\n"
        exit 1
    fi
    
    # Small delay between batches
    if [ $i -lt $NUM_BATCHES ]; then
        echo "Waiting 5 seconds before next batch..."
        sleep 5
    fi
done

# Combine all batches
echo -e "\n${BLUE}Combining all batches...${NC}"
python combine_batches.py \
    --input-files "$DATA_DIR"/batch_*.json \
    --output "$COMBINED_OUTPUT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All batches combined successfully${NC}\n"
else
    echo -e "${YELLOW}✗ Failed to combine batches${NC}\n"
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# Final summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Generation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output file: ${COMBINED_OUTPUT}"
echo -e "Log files: ${LOGS_DIR}/batch_*.log"
echo -e "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "${GREEN}========================================${NC}\n"

# Display file info
if [ -f "$COMBINED_OUTPUT" ]; then
    FILE_SIZE=$(du -h "$COMBINED_OUTPUT" | cut -f1)
    echo -e "Dataset file size: ${FILE_SIZE}"
    echo -e "\nNext steps:"
    echo -e "1. Review the dataset: cat $COMBINED_OUTPUT | jq '.metadata'"
    echo -e "2. Create train/val/test splits"
    echo -e "3. Upload to Hugging Face"
    echo -e "4. Proceed to model fine-tuning\n"
fi
