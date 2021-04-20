import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x3cf60fcb7879612ca4dE649fBbe349ab390D4798';

export default new web3.eth.Contract(jobs, address);
