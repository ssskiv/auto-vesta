ARG VESTA_IMAGE=ghcr.io/ssskiv/vesta-sim:nav2ai
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM ${VESTA_IMAGE}

COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.txt /tmp/training-requirements.txt
RUN uv venv --python /usr/bin/python3 --system-site-packages /opt/vesta-training \
    && uv pip install --python /opt/vesta-training/bin/python --no-cache -r /tmp/training-requirements.txt

ENV PATH="/opt/vesta-training/bin:${PATH}" \
    PYTHONPATH="/workspace/ros2_ws/src/offroad_nav2/tools" \
    NO_ALBUMENTATIONS_UPDATE=1
WORKDIR /workspace/training
ENTRYPOINT []
CMD ["bash"]
