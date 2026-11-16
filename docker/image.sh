#!/usr/bin/env bash
# RU: Общий выбор образа. Для прямого Compose: source docker/image.sh
# EN: Shared image selection. Before direct Compose: source docker/image.sh
export VESTA_IMAGE="${VESTA_IMAGE:-ghcr.io/ssskiv/vesta-sim:nav2ai}"

# RU: Проверяем кэш или скачиваем; pull явно обновляет образ из реестра.
# EN: Inspect the cache or download; pull explicitly refreshes a registry image.
vesta_image_acquire() {
    case "$VESTA_IMAGE" in
        -*|*[[:space:]]*)
            echo "RU: Некорректный VESTA_IMAGE | EN: Invalid VESTA_IMAGE: $VESTA_IMAGE" >&2
            return 1
            ;;
    esac
    if [ "${1:-}" = pull ] || ! docker image inspect "$VESTA_IMAGE" >/dev/null 2>&1; then
        if [ "$VESTA_IMAGE" = vesta-sim:latest ]; then
            echo "RU: Нужна локальная сборка: make image-build | EN: Build the local image: make image-build" >&2
            return 1
        fi
        docker pull "$VESTA_IMAGE" || return $?
    fi
    local metadata
    metadata="$(docker image inspect --format '{{.Os}}/{{.Architecture}} {{index .Config.Labels "vesta.nav"}}' "$VESTA_IMAGE")" || return $?
    if [ "$metadata" != 'linux/amd64 2' ]; then
        echo "RU: Несовместимый образ | EN: Incompatible image $VESTA_IMAGE: $metadata (requires linux/amd64, vesta.nav=2)" >&2
        return 1
    fi
}
