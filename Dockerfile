FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive


# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    git\
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Create virtual environment
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Install LeRobot
COPY . /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[aloha]"

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"

ENTRYPOINT ["sh", "-c", "python lerobot/scripts/eval.py --pretrained-policy-name-or-path ${POLICY_NAME} --out-dir output \"$@\"", "--"]
CMD ["eval.n_episodes=100", "eval.batch_size=1"]
# docker run -v ./output:/lerobot/output --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all  --runtime=nvidia -e POLICY_NAME=HumanoidTeam/hackathon_sim_aloha image_name eval.n_episodes=10 eval.batch_size=1