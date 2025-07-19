#!/usr/bin/env python3
"""Initialize the Screen Memory Assistant database"""
import asyncio
import os
from database import db

async def main():
    # Set the database URL for the working PostgreSQL connection (no auth needed)
    os.environ['DATABASE_URL'] = 'postgresql+asyncpg://@localhost:5432/postgres'
    
    try:
        print("🔄 Initializing database...")
        await db.initialize()
        print('✅ Database initialized successfully!')
        
        # Test the connection
        print("🔄 Testing database connection...")
        healthy = await db.health_check()
        if healthy:
            print('✅ Database health check passed!')
        else:
            print('❌ Database health check failed')
            
        # Get some stats
        print("🔄 Getting database stats...")
        stats = await db.get_stats()
        print(f'📊 Database stats: {stats}')
        
    except Exception as e:
        print(f'❌ Database error: {e}')
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    print("\n🎉 Database is ready!" if success else "\n❌ Database setup failed") 