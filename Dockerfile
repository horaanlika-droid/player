# BotHost: «Использовать собственный Dockerfile» + главный файл index.js.
# Приложению нужен только рантайм Node.js — зависимостей у сервера нет,
# поэтому образ собирается за секунды и не ломается на npm install.
FROM node:20-alpine

WORKDIR /app

# сначала манифест — слой кэшируется, пока не менялся package.json
COPY package*.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY . .

ENV NODE_ENV=production \
    PORT=8080

# приложение само читает PORT из окружения и слушает 0.0.0.0
EXPOSE 8080

# пингуем собственный /health: приложение знает свой порт из PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8080)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "index.js"]
