#!/usr/bin/env python3
"""Direct database setup for Screen Memory Assistant"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Import our models to create tables
from models import Base

async def main():
    # Use the working connection format
    engine = create_async_engine('postgresql+asyncpg://@localhost:5432/postgres')
    
    try:
        print("🔄 Creating database schema...")
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database schema created successfully!")
        
        # Test by inserting some sample data
        async with engine.begin() as conn:
            # Insert a test event
            await conn.execute(text("""
                INSERT INTO screen_events (ts, window_title, app_name, full_text, ocr_conf, created_at) 
                VALUES (NOW(), 'Test Window', 'Test App', 'Hello Screen Memory!', 95, NOW())
            """))
            
            print("✅ Sample data inserted!")
            
            # Query it back
            result = await conn.execute(text("SELECT COUNT(*) FROM screen_events"))
            count = result.scalar()
            print(f"📊 Total events in database: {count}")
            
        await engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        await engine.dispose()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print("\n🎉 Database is ready for Screen Memory Assistant!" if success else "\n❌ Database setup failed") 