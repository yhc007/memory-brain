#!/bin/bash
# Notion <-> Memory Brain 양방향 동기화
# Usage: ./notion-sync.sh [push|pull|sync]

set -e

NOTION_KEY=$(cat ~/.config/notion/api_key 2>/dev/null || echo "")
NOTION_DB_ID="2f99fba4-1978-8175-803e-e7689e1b0411"
MEMORY_BRAIN="cargo run --release -q --manifest-path /Volumes/T7/Work/memory-brain/Cargo.toml --"

if [ -z "$NOTION_KEY" ]; then
    echo "❌ Notion API key not found. Run: echo 'your_key' > ~/.config/notion/api_key"
    exit 1
fi

notion_api() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -n "$data" ]; then
        curl -s -X "$method" "https://api.notion.com/v1/$endpoint" \
            -H "Authorization: Bearer $NOTION_KEY" \
            -H "Notion-Version: 2022-06-28" \
            -H "Content-Type: application/json" \
            -d "$data"
    else
        curl -s -X "$method" "https://api.notion.com/v1/$endpoint" \
            -H "Authorization: Bearer $NOTION_KEY" \
            -H "Notion-Version: 2022-06-28"
    fi
}

# Push: memory-brain → Notion
push_to_notion() {
    echo "📤 Pushing memories to Notion..."
    
    # Get all memories from memory-brain (JSON format)
    memories=$($MEMORY_BRAIN export --format json 2>/dev/null || echo "[]")
    
    if [ "$memories" = "[]" ]; then
        echo "   No memories to push"
        return
    fi
    
    count=0
    echo "$memories" | jq -c '.[]' | while read -r memory; do
        id=$(echo "$memory" | jq -r '.id')
        content=$(echo "$memory" | jq -r '.content')
        tags=$(echo "$memory" | jq -r '.tags | join(", ")')
        mem_type=$(echo "$memory" | jq -r '.memory_type // "semantic"')
        strength=$(echo "$memory" | jq -r '.strength // 0.5')
        created=$(echo "$memory" | jq -r '.created_at // empty')
        
        # Check if already exists in Notion
        existing=$(notion_api POST "databases/$NOTION_DB_ID/query" "{\"filter\":{\"property\":\"ID\",\"rich_text\":{\"equals\":\"$id\"}}}" | jq -r '.results[0].id // empty')
        
        if [ -n "$existing" ]; then
            echo "   ⏭️ Skip (exists): ${content:0:40}..."
            continue
        fi
        
        # Build tags array
        tags_json=$(echo "$memory" | jq '[.tags[] | {name: .}]')
        
        # Create in Notion
        notion_api POST "pages" "{
            \"parent\": {\"database_id\": \"$NOTION_DB_ID\"},
            \"properties\": {
                \"Content\": {\"title\": [{\"text\": {\"content\": \"$content\"}}]},
                \"ID\": {\"rich_text\": [{\"text\": {\"content\": \"$id\"}}]},
                \"Type\": {\"select\": {\"name\": \"$mem_type\"}},
                \"Strength\": {\"number\": $strength},
                \"Synced\": {\"checkbox\": true}
            }
        }" > /dev/null
        
        echo "   ✅ Pushed: ${content:0:40}..."
        count=$((count + 1))
    done
    
    echo "📤 Push complete!"
}

# Pull: Notion → memory-brain
pull_from_notion() {
    echo "📥 Pulling memories from Notion..."
    
    # Query all unsynced items from Notion
    response=$(notion_api POST "databases/$NOTION_DB_ID/query" '{"filter":{"property":"Synced","checkbox":{"equals":false}}}')
    
    count=0
    echo "$response" | jq -c '.results[]' | while read -r page; do
        content=$(echo "$page" | jq -r '.properties.Content.title[0].plain_text // empty')
        page_id=$(echo "$page" | jq -r '.id')
        mem_type=$(echo "$page" | jq -r '.properties.Type.select.name // "semantic"')
        
        if [ -z "$content" ]; then
            continue
        fi
        
        # Store in memory-brain
        $MEMORY_BRAIN store "$content" --tags "notion,$mem_type" 2>/dev/null
        
        # Mark as synced in Notion
        notion_api PATCH "pages/$page_id" '{"properties":{"Synced":{"checkbox":true}}}' > /dev/null
        
        echo "   ✅ Pulled: ${content:0:40}..."
        count=$((count + 1))
    done
    
    echo "📥 Pull complete!"
}

# Full sync
full_sync() {
    pull_from_notion
    echo ""
    push_to_notion
}

case "${1:-sync}" in
    push)
        push_to_notion
        ;;
    pull)
        pull_from_notion
        ;;
    sync)
        full_sync
        ;;
    *)
        echo "Usage: $0 [push|pull|sync]"
        echo "  push  - memory-brain → Notion"
        echo "  pull  - Notion → memory-brain"
        echo "  sync  - Both directions (default)"
        exit 1
        ;;
esac
