from outboxml.service import run_service

def main(host="127.0.0.1",
         port=8080,
         ):
    run_service(
     host=host,
     port=port)

if __name__ == "__main__":
    main()
