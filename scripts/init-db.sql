-- Initialize WMS database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create application user with limited privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wms_app') THEN
        CREATE ROLE wms_app WITH LOGIN PASSWORD 'wms_app_pass';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE wms_db TO wms_app;
GRANT USAGE ON SCHEMA public TO wms_app;
GRANT CREATE ON SCHEMA public TO wms_app;
