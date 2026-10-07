import EmbeddedPostgres from 'embedded-postgres';
import { existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../backend');
process.loadEnvFile(path.join(root, '.env'));
const databaseDir = path.join(root, '.local/postgres');
const socketDir = path.join(root, '.local/socket');
mkdirSync(socketDir, { recursive: true });
const postgres = new EmbeddedPostgres({ databaseDir, user: 'securelens',
  password: process.env.LOCAL_POSTGRES_PASSWORD, port: 55432, persistent: true,
  authMethod: 'scram-sha-256', createPostgresUser: false,
  initdbFlags: ['--locale=C', '--encoding=UTF8'],
  postgresFlags: ['-h', '127.0.0.1', '-k', socketDir],
  onLog: () => {}, onError: message => console.error(message) });
if (!existsSync(path.join(databaseDir, 'PG_VERSION'))) await postgres.initialise();
await postgres.start();
const client = postgres.getPgClient();
await client.connect();
const found = await client.query("SELECT 1 FROM pg_database WHERE datname = 'securelens'");
await client.end();
if (!found.rowCount) await postgres.createDatabase('securelens');
console.log('Local PostgreSQL is listening on 127.0.0.1:55432. Press Ctrl+C to stop.');
let stopping = false;
async function stop() { if (stopping) return; stopping = true; await postgres.stop(); process.exit(0); }
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
