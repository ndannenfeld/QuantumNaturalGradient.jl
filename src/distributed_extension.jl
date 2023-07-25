function remotecall_eval_async(m::Module, procs, ex)
    
    run_locally = 0
    for pid in procs
        if pid == myid()
            run_locally += 1
        else
            Distributed.@async_unwrap remotecall_wait(Core.eval, pid, m, ex)
        end
    end
    yield() # ensure that the remotecalls have had a chance to start

    # execute locally last as we do not want local execution to block serialization
    # of the request to remote nodes.
    for _ in 1:run_locally
        @async Core.eval(m, ex)
    end

    return nothing
end

macro everywhere_async(ex)
    procs = GlobalRef(@__MODULE__, :procs)
    return esc(:($(SRMPS).@everywhere_async $procs() $ex))
end

macro everywhere_async(procs, ex)
    imps = Distributed.extract_imports(ex)
    return quote
        $(isempty(imps) ? nothing : Expr(:toplevel, imps...)) # run imports locally first
        let ex = Expr(:toplevel, :(task_local_storage()[:SOURCE_PATH] = $(get(task_local_storage(), :SOURCE_PATH, nothing))), $(esc(Expr(:quote, ex)))),
            procs = $(esc(procs))
            remotecall_eval_async(Main, procs, ex)
        end
    end
end

macro addprocs_and_everywhere(num_procs, ex, max_add_num_procs=numprocs, verbose=false)
    return quote
        num_procs = $(esc(num_procs))
        max_add_num_procs = $(esc(max_add_num_procs))
        @eval $ex
        while length(procs()) < num_procs
            lw = length(procs())
            
            ad = min(num_procs - lw, max_add_num_procs)
            added_workers = addprocs(ad)
            if $verbose
                @time @everywhere added_workers $ex
                println("Total: ", length(procs()) , " - Added ", ad, " procs")
            else
                @everywhere added_workers @eval $ex
            end
        end
    end   
end