-- Supabase exposes the public schema through its REST API. Enable Row Level Security on every table
-- and allow only read access for the anon/authenticated roles (the dashboard). Collectors write through
-- DATABASE_URL as the postgres role, which bypasses RLS.
DO $$
DECLARE t text;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS read_only ON public.%I', t);
    EXECUTE format('CREATE POLICY read_only ON public.%I FOR SELECT TO anon, authenticated USING (true)', t);
  END LOOP;
END $$;
