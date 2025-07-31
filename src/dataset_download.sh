V7W_JSON_URL="http://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
V7W_IMAGE_URL="http://vision.stanford.edu/yukezhu/visual7w_images.zip"
DOWNLOAD_DIR="datasets/visual7w"
V7W_JSON_ZIP="v7w_telling.zip"
V7W_IMAGES_ZIP="visual7w_images.zip"

# 디렉토리 생성
echo "Creating directory $DOWNLOAD_DIR"
mkdir -p "$DOWNLOAD_DIR"

# 다운로드 메시지 출력
echo "Downloading Visual 7w dataset..."

if wget -c $V7W_JSON_URL -O "$V7W_JSON_ZIP"; then
    echo "Download 1 successful."
else
    echo "Error: Download failed. Please check the URL or your network connection."
    exit 1
fi

if wget -c $V7W_IMAGE_URL -O "$V7W_IMAGES_ZIP"; then
    echo "Download 2 successful."
else
    echo "Error: Download failed. Please check the URL or your network connection."
    exit 1
fi

# 압축 해제
echo "Unzipping dataset..."
unzip "$V7W_JSON_ZIP" -d "$DOWNLOAD_DIR"
unzip "$V7W_IMAGES_ZIP" -d "$DOWNLOAD_DIR"

# 압축 파일 삭제
rm "$V7W_JSON_ZIP"
rm "$V7W_IMAGES_ZIP"

echo "visual 7w dataset download success."

# 압축 해제
echo "Unzipping DACON dataset..."
unzip "eg/open.zip" -d "eg/"
