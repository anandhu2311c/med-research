FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Serve the application
EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]